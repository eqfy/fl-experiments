# -*- coding: utf-8 -*-

from enum import Enum
import traceback
from utils.tensor_evaluator import TensorEvaluator
from fl_aggregator_libs import *
from random import Random
from resource_manager import ResourceManager
from communication.channelcontext import ExecutorConnections
from response import BasicResponse

import job_api_pb2_grpc
import job_api_pb2
import grpc
import io
import torch
import pickle
from torch.utils.tensorboard import SummaryWriter
import threading
import round_evaluator

def check_sparsification_ratio(global_g_list):
    worker_number = len(global_g_list)
    spar_ratio = 0.

    total_param = 0
    for g_idx, g_param in enumerate(global_g_list[0]):
        total_param += len(torch.flatten(global_g_list[0][g_idx]))

    for i in range(worker_number):
        non_zero_param = 0
        for g_idx, g_param in enumerate(global_g_list[i]):
            mask = g_param != 0.
            # print(mask)
            non_zero_param += float(torch.sum(mask))

        spar_ratio += (non_zero_param / total_param) / worker_number

    return spar_ratio

class SamplingStrategy(Enum):
    UNIFORM = 0
    STICKY = 1

class FLMethod(Enum):
    FedAvg = 0
    STC = 1
    APF = 2
    FedDC = 3
    FedDCPrefetch = 4

class Aggregator(object):
    """This centralized aggregator collects training/testing feedbacks from executors"""
    def __init__(self, args):
        logging.info(f"Job args {args}")

        self.args = args
        self.device = args.cuda_device if args.use_cuda else torch.device('cpu')

        # ======== env information ========
        self.this_rank = 0
        self.global_virtual_clock = 0.
        self.round_duration = 0.
        self.resource_manager = ResourceManager()
        self.client_manager = self.init_client_manager(args=args)
        self.tensor_evaluator = TensorEvaluator(epochs=self.args.epochs)
        self.worker_pool_limit = 40
        self.dataset_total_worker = 2800 # to distinguish between self.args.total_worker which is the total worker in a round
        self.round_total_worker = args.total_worker
        self.change_num = 7

        # ======== model and data ========
        self.last_update_index = None
        self.mask_record_list = []

        # list of parameters in model.parameters()
        self.model_in_update = 0
        self.update_lock = threading.Lock()
        self.last_global_model = []
        self.last_global_gradient = []
        self.model_state_dict = None
        self.compressed_gradient = None
        self.last_compressed_gradient = None
        self.overlap_gradient = None
        # ===== updates 

        # ======== channels ========
        self.executors = None

        # event queue of its own functions
        self.event_queue = collections.deque()
        self.client_result_queue = []

        # ======== runtime information ========
        self.tasks_round = 0
        self.sampled_participants = []

        self.round_stragglers = []
        self.model_size = 0.

        self.collate_fn = None
        self.task = args.task
        self.epoch = 0
        # contain every client group from the past, the current client group, and scheduled future client group (indexed by epoch)
        self.client_groups = [] 
        # contains every worker pool, including the next worker pool for generating the next client group
        self.worker_pools = []
        self.curr_worker_pool = []
        self.cur_change_num = 7

        # ======== experiment configs =========
        self.sampling_strategy = SamplingStrategy.STICKY
        self.fl_method = FLMethod.FedDCPrefetch

        # ======== scheduling ========
        self.max_prefetch_round = 5
        self.sticky_model_update_size = 0.
        self.compressed_gradient_size = 0.
        self.mask_model_size = 0.
        self.total_mask_ratio = 0.2  # = shared_mask + local_mask
        self.shared_mask_ratio = 0.18
        self.client_prefetch_rounds = [] # a list of 
        self.round_durations = []

        self.update_bitmap_size = 0.
        self.local_mask_size = 0.

        self.start_run_time = time.time()
        self.client_conf = {}

        self.stats_util_accumulator = []
        self.loss_accumulator = []
        self.client_training_results = []

        # number of registered executors
        self.registered_executor_info = set()
        self.test_result_accumulator = []
        self.testing_history = {'data_set': args.data_set, 'model': args.model, 'sample_mode': args.sample_mode,
                        'gradient_policy': args.gradient_policy, 'task': args.task, 'perf': collections.OrderedDict()}

        self.log_writer = SummaryWriter(log_dir=logDir)
        self.temp_grad_path = os.path.join(logDir, '../executor/grad_0.pth.tar')
        # ======== Task specific ============
        self.imdb = None           # object detection

        # ======== debugging ========
        self.sticky_total_prob = 0.


    def setup_env(self):
        self.setup_seed(seed=self.this_rank)

        # set up device
        if self.args.use_cuda and self.device == None:
            for i in range(torch.cuda.device_count()):
                try:
                    self.device = torch.device('cuda:'+str(i))
                    torch.cuda.set_device(i)
                    _ = torch.rand(1).to(device=self.device)
                    logging.info(f'End up with cuda device ({self.device})')
                    break
                except Exception as e:
                    assert i != torch.cuda.device_count()-1, 'Can not find available GPUs'

        self.init_control_communication()
        self.init_data_communication()
        self.optimizer = ServerOptimizer(self.args.gradient_policy, self.args, self.device)

    def setup_seed(self, seed=1):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def init_control_communication(self):
        # Create communication channel between aggregator and worker
        # This channel serves control messages
        logging.info(f"Initiating control plane communication ...")
        self.executors = ExecutorConnections(self.args.executor_configs, self.args.base_port)


    def init_data_communication(self):
        """For jumbo traffics (e.g., training results).
        """
        pass

    def init_model(self):
        """Load model"""
        if self.args.task == "detection":
            cfg_from_file(self.args.cfg_file)
            np.random.seed(self.cfg.RNG_SEED)
            self.imdb, _, _, _ = combined_roidb("voc_2007_test", ['DATA_DIR', self.args.data_dir], server=True)

        return init_model()

    def init_client_manager(self, args):
        """
            Currently we implement two client managers:
            1. Random client sampler
                - it selects participants randomly in each round
                - [Ref]: https://arxiv.org/abs/1902.01046
            2. Oort sampler
                - Oort prioritizes the use of those clients who have both data that offers the greatest utility
                  in improving model accuracy and the capability to run training quickly.
                - [Ref]: https://www.usenix.org/conference/osdi21/presentation/lai
        """

        # sample_mode: random or kuiper
        client_manager = clientManager(args.sample_mode, args=args)

        return client_manager

    def load_client_profile(self, file_path):
        # load client profiles
        global_client_profile = {}
        if os.path.exists(file_path):
            with open(file_path, 'rb') as fin:
                # {clientId: [computer, bandwidth]}
                global_client_profile = pickle.load(fin)

        return global_client_profile

    def init_round_evaluator(self):
        """ 
        Initialize round evaluator to keep track of bandwidth and durations
        """
        self.round_evaluator = round_evaluator.RoundEvaluator()

    def executor_info_handler(self, executorId, info):

        self.registered_executor_info.add(executorId)
        logging.info(f"Received executor {executorId} information, {len(self.registered_executor_info)}/{len(self.executors)}")
        # have collected all executors
        # In this simulation, we run data split on each worker, so collecting info from one executor is enough
        # Waiting for data information from executors, or timeout

        if len(self.registered_executor_info) == len(self.executors):

            clientId = 1
            logging.info(f"Loading {len(info['size'])} client traces ...")

            self.last_update_index = [0 for i in range(len(info['size']) + 1)]
            self.client_prefetch_rounds = [0 for _ in range(len(info['size']) + 1)]
            for _size in info['size']:
                # since the worker rankId starts from 1, we also configure the initial dataId as 1
                mapped_id = clientId%len(self.client_profiles) if len(self.client_profiles) > 0 else 1
                systemProfile = self.client_profiles.get(mapped_id, {'computation': 1.0, 'communication':1.0})
                self.client_manager.registerClient(executorId, clientId, size=_size, speed=systemProfile)
                self.client_manager.registerDuration(clientId, batch_size=self.args.batch_size,
                    upload_epoch=self.args.local_steps, upload_size=self.model_size, download_size=self.model_size)
                clientId += 1
            # partition the client into different groups - 10 different groups
            self.client_manager.partitionClient(10)

            logging.info("Info of all feasible clients {}".format(self.client_manager.getDataInfo()))

            # start to sample clients
            self.round_completion_handler()


    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        """We try to remove dummy events as much as possible, by removing the stragglers/offline clients in overcommitment"""

        sampledClientsReal = []
        completionTimes = []
        completed_client_clock = {}
        # 1. remove dummy clients that are not available to the end of training
        for client_to_run in sampled_clients:
            client_is_sticky = client_to_run not in self.client_manager.picked_changes
            client_cfg = self.client_conf.get(client_to_run, self.args)

            bw, exe_cost = 0, {}
            if self.fl_method == FLMethod.FedAvg:
                ul_size = self.model_size
                dl_size = self.model_size
                exe_cost = self.client_manager.getCompletionTime(client_to_run, batch_size=client_cfg.batch_size,
                                                         upload_epoch=client_cfg.local_steps, upload_size=ul_size, download_size=dl_size)
                self.round_evaluator.recordClient(client_to_run, dl_size, ul_size, exe_cost)
            elif self.fl_method == FLMethod.STC:
                pass # TODO
            elif self.fl_method == FLMethod.APF:
                pass # TODO
            elif self.fl_method == FLMethod.FedDC:
                l = self.last_update_index[client_to_run]
                r = self.epoch - 1
                logging.info(f"{l} and {r}")

                downstream_update_ratio = self.tensor_evaluator.check_model_update_overhead(l, r, self.model, self.mask_record_list, self.device)
                if self.epoch == 1:
                    downstream_update_ratio = 1.

                ul_size = self.total_mask_ratio * self.model_size + min((self.total_mask_ratio - self.shared_mask_ratio) * self.model_size, self.update_bitmap_size)
                dl_size = self.update_bitmap_size + self.model_size * downstream_update_ratio
                
                exe_cost = self.client_manager.getCompletionTime(client_to_run, batch_size=client_cfg.batch_size,
                                                         upload_epoch=client_cfg.local_steps, upload_size=ul_size, download_size=dl_size)
                self.round_evaluator.recordClient(client_to_run, dl_size, ul_size, exe_cost)

            elif self.fl_method == FLMethod.FedDCPrefetch:
                prefetch_round = 0
                can_fully_prefetch = False
                prefetch_size = 0

                # Calculate backwards to see if client can finish prefetching in max_prefetch_round
                for i in range(1, min(self.max_prefetch_round, self.epoch - 1)):
                    l, r = self.last_update_index[client_to_run], self.epoch - 1 - i
                    if l > r: # This case usually happens when the client participated in training recently
                        break

                    round_durations = self.round_evaluator.getDurationsAggregated()[max(0, self.epoch - 1 - self.max_prefetch_round):self.epoch - 1 - i]
                    min_round_duration = min(round_durations)

                    logging.info(f"Estimate prefetch, l{l} and r{r}, round_durations {round_durations}, preround {i}\nall round_durations{self.round_evaluator.getDurationsAggregated()}\ni {i} client_id{client_to_run}")
                    
                    prefetch_downstream_update_ratio = self.tensor_evaluator.check_model_update_overhead(l, r, self.model, self.mask_record_list, self.device)
                    prefetch_size = self.get_prefetch_dl_size(downstream_overhead_ratio=prefetch_downstream_update_ratio, mask_ratio=self.shared_mask_ratio)

                    temp_pre_round = self.client_manager.getDownloadTime(client_to_run, prefetch_size) / min_round_duration
                    logging.info(f"temp pre round {temp_pre_round}")

                    if temp_pre_round <= i:
                        can_fully_prefetch = True
                        prefetch_round = i
                    

                ul_size = self.total_mask_ratio * self.model_size + min((self.total_mask_ratio - self.shared_mask_ratio) * self.model_size, self.update_bitmap_size)

                if can_fully_prefetch:
                    l, r  = self.epoch - 1 - prefetch_round, self.epoch - 1
                    logging.info(f"After prefetch, l{l} and r{r}")
                    downstream_update_ratio = self.tensor_evaluator.check_model_update_overhead(l, r, self.model, self.mask_record_list, self.device)
                    dl_size = self.update_bitmap_size + self.model_size * downstream_update_ratio
                    exe_cost = self.client_manager.getCompletionTime(client_to_run, batch_size=client_cfg.batch_size,
                                                         upload_epoch=client_cfg.local_steps, upload_size=ul_size, download_size=dl_size)
                    self.round_evaluator.recordClient(client_to_run, dl_size, ul_size, exe_cost, prefetch_dl_size=prefetch_size)
                else:
                    l, r = self.last_update_index[client_to_run], self.epoch - 1
                    logging.info(f"Unable to prefetch, l{l} and r{r}")
                    downstream_update_ratio = self.tensor_evaluator.check_model_update_overhead(l, r, self.model, self.mask_record_list, self.device)
                    if self.epoch == 1:
                        downstream_update_ratio = 1
                    dl_size = self.update_bitmap_size + self.model_size * downstream_update_ratio
                    exe_cost = self.client_manager.getCompletionTime(client_to_run, batch_size=client_cfg.batch_size,
                                                         upload_epoch=client_cfg.local_steps, upload_size=ul_size, download_size=dl_size)
                    self.round_evaluator.recordClient(client_to_run, dl_size, ul_size, exe_cost)

            roundDuration = exe_cost['computation'] + exe_cost['downstream'] + exe_cost['upstream']
            # logging.info(f"client {client_to_run} bw {bw}  exec_cost {exe_cost} duration {roundDuration}")

            # if the client is not active by the time of collection, we consider it is lost in this round
            if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock):
                sampledClientsReal.append(client_to_run)
                completionTimes.append(roundDuration)
                completed_client_clock[client_to_run] = exe_cost
                self.last_update_index[client_to_run] = self.epoch - 1

        num_clients_to_collect = min(num_clients_to_collect, len(completionTimes))
        # 2. get the top-k completions to remove stragglers
        sortedWorkersByCompletion = sorted(range(len(completionTimes)), key=lambda k:completionTimes[k])
        clients_to_run, dummy_clients, round_duration, client_completion_times = [],[],0,[]

        if False: # if self.sampling_strategy == SamplingStrategy.STICKY:
            """
            UNUSED
            Maintains k sticky clients and (n - k) new clients.
            Does the above by taking the top k clients from the sticky group and the top (n-k) new clients
            """
            sticky_worker_index = []
            new_worker_index = []
            top_k_index = []
            dummy_index = []

            for i in sortedWorkersByCompletion:
                if sampledClientsReal[i] in self.client_manager.picked_changes:
                    new_worker_index.append(i)
                else:
                    sticky_worker_index.append(i)

            sticky_worker_index = sticky_worker_index[:num_clients_to_collect - 7]
            if self.epoch <= 1:
                top_k_index.extend(new_worker_index[:num_clients_to_collect])
            else:
                top_k_index.extend(new_worker_index[:7])
            
            top_k_index.extend(sticky_worker_index)

            round_duration = completionTimes[new_worker_index[-1]]
            if len(sticky_worker_index) > 0:
                round_duration = max(round_duration, completionTimes[sticky_worker_index[-1]])

            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            if self.epoch <= 1:
                dummy_index = new_worker_index[num_clients_to_collect:]
            else: 
                dummy_index.extend(sticky_worker_index[num_clients_to_collect - 7:])
                dummy_index.extend(new_worker_index[7:])
                
            dummy_clients = [sampledClientsReal[k] for k in dummy_index]            
            client_completion_times = [completionTimes[i] for i in top_k_index]

            
        else:
            top_k_index = sortedWorkersByCompletion[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            dummy_clients = [sampledClientsReal[k] for k in sortedWorkersByCompletion[num_clients_to_collect:]]
            round_duration = completionTimes[top_k_index[-1]]
            completionTimes.sort()

            client_completion_times = completionTimes[:num_clients_to_collect]

            new_client_count = 0
            for client in clients_to_run:
                if client in self.client_manager.picked_changes:
                    new_client_count += 1
            self.cur_change_num = new_client_count
        
        self.round_evaluator.recordRoundCompletion(clients_to_run, dummy_clients, round_duration)
        return clients_to_run, dummy_clients, completed_client_clock, round_duration, client_completion_times

    def get_prefetch_dl_size(self,downstream_overhead_ratio=1.0, mask_ratio=0.):
        return min(self.model_size * (1 - mask_ratio), downstream_overhead_ratio * self.model_size + self.update_bitmap_size)

    def run(self):
        self.setup_env()
        self.model = self.init_model()
        self.save_last_param()

        # --- initialize mask ---
        self.mask_model = []
        for idx, param in enumerate(self.model.state_dict().values()):
            # self.mask_model.state_dict()[idx] = torch.zeros_like(param, dtype=torch.bool).to(dtype=torch.bool)
            self.mask_model.append(torch.zeros_like(param, dtype=torch.bool).to(dtype=torch.bool))
        # for idx, param in enumerate(self.model.state_dict().values()):
        #     logging.info(f"dtype: {self.mask_model[idx].dtype}")
        logging.info("Initialize mask model")

        self.model_size = sys.getsizeof(pickle.dumps(self.model))/1024.0*8. # kbits
        self.mask_model_size = sys.getsizeof(pickle.dumps(self.mask_model))/1024.*8. # kbits
        self.update_bitmap_size = self.model_size / 32
        self.client_profiles = self.load_client_profile(file_path=self.args.device_conf_file)
        self.init_round_evaluator()
        self.event_monitor()


    def select_participants(self, select_num_participants, overcommitment=1.2, change_num=7):
        return sorted(self.client_manager.resampleClients(int(select_num_participants*overcommitment), cur_time=self.global_virtual_clock, K=100, change_num=change_num))
        # return sorted(self.client_manager.resampleClients_uniform(int(select_num_participants*overcommitment), cur_time=self.global_virtual_clock))

    def select_future_participants(self, presample_round=5, overcommitment=1.2, change_num=7, worker_pool_limit=40.):
        """ 
        In epoch 1, sample the first <presample_round> rounds, in later epochs, sample only the <presample_round> round 
        """
        group_size = self.round_total_worker
        if self.epoch == 1:
            tmp_epoch = 1
            # worker_pool = []
            # self.worker_pools.append(worker_pool)
            for _ in range(presample_round + 1):
                new_participants = sorted(self.select_participants(group_size, overcommitment, change_num))
                # new_participants = sorted(self.client_manager.resampleClients(int(group_size*overcommitment), worker_limit=worker_pool_limit,cur_time=self.global_virtual_clock, worker_pool=worker_pool, change_num=self.change_num))
                self.client_groups.append(new_participants)

                tmp_epoch += 1
                # worker_pool = self.get_scheduled_worker_pool(tmp_epoch, worker_limit=worker_pool_limit)
                # self.worker_pools.append(worker_pool)
        else:
            # worker_pool = self.worker_pools[self.epoch + presample_round - 1]
            new_participants = sorted(self.select_participants(group_size, overcommitment, change_num))
            # new_participants = sorted(self.client_manager.resampleClients(int(group_size*overcommitment), worker_limit=worker_pool_limit,cur_time=self.global_virtual_clock, worker_pool=worker_pool, change_num=self.change_num))
            self.client_groups.append(new_participants)
            # self.worker_pools.append(self.get_scheduled_worker_pool(self.epoch + presample_round,  worker_limit=worker_pool_limit))


    # def get_scheduled_worker_pool(self, epoch, worker_limit):
    #     """Generates the worker pool for <epoch + 1>"""
    #     worker_pool = []
    #     for i in range(0, epoch - 1):
    #         idx = len(self.client_groups) - 1 - i
    #         for worker in self.client_groups[idx]:
    #             if worker not in worker_pool:
    #                 worker_pool.append(worker)
    #             if len(worker_pool) >= worker_limit:
    #                 break
        
    #         if len(worker_pool) >= worker_limit:
    #             break
    #     # logging.info(f"{epoch - 1} {len(self.client_groups) - 1}\nclient groups{self.client_groups} \nworker pool {worker_pool}")
    #     return worker_pool

    def update_gradient_handler(self, gradients):
        """Update last round global gradients"""
        # serialized_data = pickle.dumps(gradients.to(device='cpu'))
        # gradients = pickle.loads(serialized_data)
        
        # Dump latest model to disk
        with open(self.temp_grad_path, 'wb') as grad_out:
            pickle.dump(gradients, grad_out)

    def client_completion_handler(self, results):
        """We may need to keep all updates from clients, if so, we need to append results to the cache"""
        # Format:
        #       -results = {'clientId':clientId, 'update_weight': model_param, 'moving_loss': epoch_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}
        
        # if self.args.gradient_policy in ['q-fedavg']:
        #     self.client_training_results.append(results)
        
        if True:
            self.client_training_results.append(results)

        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results['utility'])
        self.loss_accumulator.append(results['moving_loss'])

        self.client_manager.registerScore(results['clientId'], results['utility'], auxi=math.sqrt(results['moving_loss']),
            time_stamp=self.epoch,
            duration=self.virtual_client_clock[results['clientId']]['computation']+self.virtual_client_clock[results['clientId']]['downstream'] + self.virtual_client_clock[results['clientId']]['upstream']
        )

        device = self.device
        """
            [FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data".
            H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
        """
        # Start to take the average of updates, and we do not keep updates to save memory
        # Importance of each update is 1/#_of_participants
        # importance = 1./self.tasks_round

        self.update_lock.acquire()

        # ================== Aggregate weights ======================

        self.model_in_update += 1

        # if self.model_in_update == 1:
        #     self.model_state_dict = self.model.state_dict()
        #     for idx, param in enumerate(self.model_state_dict.values()):
        #         param.data = (torch.from_numpy(results['update_weight'][idx]).to(device=device))
        # else:
        #     for idx, param in enumerate(self.model_state_dict.values()):
        #        param.data += (torch.from_numpy(results['update_weight'][idx]).to(device=device))

        # if self.model_in_update == self.tasks_round:
        #     for idx, param in enumerate(self.model_state_dict.values()):
        #         param.data = (param.data/float(self.tasks_round)).to(dtype=param.data.dtype)

        #     self.model.load_state_dict(self.model_state_dict)

        # === calculate mask info ===
        l = self.last_update_index[results['clientId']]
        r = self.epoch - 1
        # logging.info(f"{l} and {r}")

        # downstream_ratio = check_model_update_overhead(l, r, self.model, self.mask_record_list, device)
        # downstream_ratio_0 = check_model_update_overhead(0, r, self.model, self.mask_record_list, device)

        # logging.info(f"round: {r} client: {results['clientId']} downstream ratio: {downstream_ratio:.6f}")
        # logging.info(f"round: {r} client: {results['clientId']} downstream_0 ratio: {downstream_ratio_0:.6f}")
        
        # Initialize compressed gradients
        if self.model_in_update == 1:
            self.compressed_gradient = [torch.zeros_like(param.data).to(device=device).to(dtype=torch.float32) for param in self.model.state_dict().values()]
            # self.overlap_gradient = [torch.zeros_like(param.data).to(device=device).to(dtype=torch.bool) | True for param in self.model.state_dict().values()]

        client_id = int(results['clientId']) 
        client_is_sticky = not(client_id in self.client_manager.picked_changes)

        prob = 0
        if self.sampling_strategy == SamplingStrategy.STICKY:
            if self.epoch <= 1:
                prob = (1.0 / float(self.tasks_round))
            elif client_is_sticky:
                prob = (1.0 / float(self.dataset_total_worker)) * (1.0 / ((float(self.tasks_round) - float(self.cur_change_num)) / float(self.tasks_round)))
            else:
                prob = (1.0 / float(self.dataset_total_worker)) * (1.0 / (float(self.cur_change_num) / (float(self.dataset_total_worker) - float(self.tasks_round))))
            
            # For debugging purposes, something is wrong if sticky_total_prob != 1 and all clients have finished
            self.sticky_total_prob += prob
            logging.info(f"client {results['clientId']} has prob {prob} round total prob {self.sticky_total_prob} is sticky {client_is_sticky}\n \
                round worker count {self.tasks_round}\t change_num {self.cur_change_num}\t data set {self.dataset_total_worker}")
            if self.model_in_update == self.tasks_round:
                self.sticky_total_prob = 0
        else: # Default is uniform sampling
            prob = (1. / float(self.tasks_round))
        
        for idx, param in enumerate(self.model.state_dict().values()):
            self.compressed_gradient[idx] += (torch.from_numpy(results['update_gradient'][idx]).to(device=device) * prob) 
            # self.overlap_gradient[idx] &= (torch.from_numpy(results['update_gradient'][idx]).to(device=device).to(dtype=torch.bool))

        # All clients are done
        if self.model_in_update == self.tasks_round:
            
            # logging.info(f"server round: {self.epoch}")

            keys = [] 
            for idx, key in enumerate(self.model.state_dict()):
                keys.append(key)

            # --- apply masking ---
            from topk import TopKCompressor
            # compressor_tot = TopKCompressor(compress_ratio=1.0)            
            # compressor_chg = TopKCompressor(compress_ratio=0.20)         
            compressor_tot = TopKCompressor(compress_ratio=self.total_mask_ratio)            
            compressor_chg = TopKCompressor(compress_ratio=self.shared_mask_ratio)    # Shared mask ratio      
            
            
            # if self.epoch >= 0:
            for idx, param in enumerate(self.model.state_dict().values()):
                # self.overlap_gradient[idx] = self.overlap_gradient[idx].to(dtype=torch.float32)

                # --- STC ---
                if self.epoch % 1 == 0:
                    self.compressed_gradient[idx], ctx_tmp = compressor_tot.compress(
                        self.compressed_gradient[idx])
                    
                    self.compressed_gradient[idx] = compressor_tot.decompress(self.compressed_gradient[idx], ctx_tmp)
                else:
                    # shared + local mask
                    # self.compressed_gradient[idx][self.mask_model[idx] != True] = 0.0
                    max_value = float(self.compressed_gradient[idx].abs().max())
                    update_mask = self.compressed_gradient[idx].clone().detach()
                    update_mask[self.mask_model[idx] == True] = max_value
                    update_mask, ctx_tmp = compressor_tot.compress(update_mask)
                    update_mask = compressor_tot.decompress(update_mask, ctx_tmp)
                    update_mask = update_mask.to(torch.bool)

                    self.compressed_gradient[idx][update_mask != True] = 0.0
                

                # self.compressed_gradient[idx], ctx_tmp = compressor_tot.compress(
                #         self.compressed_gradient[idx])
                    
                # self.compressed_gradient[idx] = compressor_tot.decompress(self.compressed_gradient[idx], ctx_tmp)
            
            # avg_waste_ratio = 0.0
            # for i in range(len(self.client_training_results)):
            #     overlap_gradient = []
            #     for idx, param in enumerate(self.model.state_dict().values()):
            #         tmp = torch.from_numpy(self.client_training_results[i]['update_gradient'][idx]).to(dtype=torch.bool).to(device=device)
            #         tmp_mask = self.compressed_gradient[idx].clone().detach().to(dtype=torch.bool).to(device=device)
                
            #         overlap_gradient.append((tmp & tmp_mask).to(dtype=torch.float32))
                
            #     overlap_ratio = check_sparsification_ratio([overlap_gradient])
            #     waste_ratio = 0.30 - overlap_ratio
            #     logging.info(f"waste sparsification: {waste_ratio} id: {self.client_training_results[i]['clientId']}")
            #     avg_waste_ratio += waste_ratio / len(self.client_training_results)
            
            # logging.info(f"waste average sparsification: {avg_waste_ratio}")

            # --- update shared mask ---
            for idx, param in enumerate(self.model.state_dict().values()):
                # shared mask
                determined_mask = self.compressed_gradient[idx].clone().detach()
                determined_mask, ctx_tmp = compressor_chg.compress(determined_mask)
                determined_mask = compressor_chg.decompress(determined_mask, ctx_tmp)

                self.mask_model[idx] = determined_mask.to(torch.bool)

            
            spar_ratio = check_sparsification_ratio([self.compressed_gradient])
            mask_ratio = check_sparsification_ratio([self.mask_model])
            # ovlp_ratio = check_sparsification_ratio([self.overlap_gradient])
            logging.info(f"Gradients sparsification: {spar_ratio}")
            logging.info(f"Mask sparsification: {mask_ratio}")
            # logging.info(f"Overlap sparsification: {ovlp_ratio}")
            
            self.update_gradient_handler(self.compressed_gradient)
            self.last_global_gradient = self.compressed_gradient

            self.model_state_dict = self.model.state_dict()
            for idx, param in enumerate(self.model_state_dict.values()):
                param.data = param.data.to(device=device).to(dtype=torch.float32) - self.compressed_gradient[idx]

            # if self.epoch > 1:
            #     overlap_gradient = []
            #     for idx, param in enumerate(self.model.state_dict().values()):
            #         if (('num_batches_tracked' in keys[idx]) or ('running' in keys[idx])):
            #             continue
            #         tmp_now = self.last_compressed_gradient[idx].clone().detach().to(dtype=torch.bool).to(device=device)
            #         tmp_last = self.compressed_gradient[idx].clone().detach().to(dtype=torch.bool).to(device=device)
            #         overlap_gradient.append((tmp_now & tmp_last).to(dtype=torch.float32))
            #         # logging.info(f"Layer overlap sparsification (last round): {check_sparsification_ratio([overlap_gradient[-1]])} id: {idx} {keys[idx]}")
            #     overlap_ratio = check_sparsification_ratio([overlap_gradient])
            #     logging.info(f"Overlap sparsification (last round): {overlap_ratio}")
                
            self.last_compressed_gradient = self.compressed_gradient

            self.model.load_state_dict(self.model_state_dict)
                
            # == update mask ===

            mask_list = []
            for p_idx, key in enumerate(self.model.state_dict().keys()):

                mask = (self.compressed_gradient[p_idx] != 0).to(device)
                # if self.epoch <= 11:
                #     mask = torch.zeros_like(mask).to(device)
                mask_list.append(mask)

            self.mask_record_list.append(mask_list)
    
        self.update_lock.release()

    def save_last_param(self):
        self.last_global_model = [param.data.clone() for param in self.model.parameters()]


    def round_weight_handler(self, last_model, current_model):
        if self.epoch > 1:
            self.optimizer.update_round_gradient(last_model, current_model, self.model)

    def round_completion_handler(self):
        self.global_virtual_clock += self.round_duration
        self.epoch += 1

        if self.epoch % self.args.decay_epoch == 0:
            self.args.learning_rate = max(self.args.learning_rate*self.args.decay_factor, self.args.min_learning_rate)

        # handle the global update w/ current and last
        self.round_weight_handler(self.last_global_model, [param.data.clone() for param in self.model.parameters()])

        avgUtilLastEpoch = sum(self.stats_util_accumulator)/max(1, len(self.stats_util_accumulator))
        # assign avg reward to explored, but not ran workers
        for clientId in self.round_stragglers:
            self.client_manager.registerScore(clientId, avgUtilLastEpoch,
                    time_stamp=self.epoch,
                    duration=self.virtual_client_clock[clientId]['computation']+self.virtual_client_clock[clientId]['downstream']+self.virtual_client_clock[clientId]['upstream'],
                    success=False)

        avg_loss = sum(self.loss_accumulator)/max(1, len(self.loss_accumulator))
        logging.info(f"Wall clock: {round(self.global_virtual_clock)} s, Epoch: {self.epoch}, Planned participants: " + \
            f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}")

        # dump round completion information to tensorboard
        if len(self.loss_accumulator):
            self.log_writer.add_scalar('Train/round_to_loss', avg_loss, self.epoch)

            self.log_writer.add_scalar('FAR/time_to_train_loss (min)', avg_loss, self.global_virtual_clock/60.)
            self.log_writer.add_scalar('FAR/round_duration (min)', self.round_duration/60., self.epoch)
            self.log_writer.add_histogram('FAR/client_duration (min)', self.flatten_client_duration, self.epoch)

        if self.epoch > 1:
            logging.info(f"Cumulative bandwidth usage:\n \
            total (excluding overcommit): {self.round_evaluator.getTotalBandwidth():.2f} kbit\n \
            total (including overcommit): {self.round_evaluator.getTotalBandwidth() + self.round_evaluator.getTotalBandwidthOvercommit():.2f} kbit\n \
            downstream: {self.round_evaluator.getTotalBandwidthDl():.2f} kbit\tupstream: {self.round_evaluator.getTotalBandwidthUl():.2f} kbit\tprefetch: {self.round_evaluator.getTotalBandwidthSchedule():.2f} kbit\tovercommit: {self.round_evaluator.getTotalBandwidthOvercommit():.2f} kbit")
            logging.info(f"Cumulative round durations:\n \
            total: {self.round_evaluator.getTotalDuration():.2f} s\n \
            avg_dl: {self.round_evaluator.getAvgDurationDl():.2f} s\n \
            avg_ul: {self.round_evaluator.getAvgDurationUl():.2f} s\n \
            avg_compute: {self.round_evaluator.getAvgDurationCompute():.2f} s\n")
            self.round_evaluator.startNewRound()

        # Sampling with prefetch
        if self.sampling_strategy == SamplingStrategy.STICKY:
            self.select_future_participants(presample_round=self.max_prefetch_round, overcommitment=self.args.overcommitment, change_num=self.change_num)
            # self.curr_worker_pool = self.worker_pools[self.epoch - 1]
            self.sampled_participants = self.client_groups[self.epoch - 1]
            logging.info(f"Epoch: {self.epoch}\tClient group size {len(self.client_groups)}\n\nCurrent client group/sampled_participant {self.sampled_participants}")
        else: # Default is uniform sampling
            num_selected = self.round_total_worker
            self.sampled_participants = self.select_participants(
                            select_num_participants=num_selected, overcommitment=self.args.overcommitment)
        
        # Only determines which client to run, the virtual_client_clock, round_duration, flatten_client_duration are unused
        clientsToRun, round_stragglers, virtual_client_clock, round_duration, flatten_client_duration = self.tictak_client_tasks(
                        self.sampled_participants, self.round_total_worker)
        self.cur_client_group = clientsToRun
        self.round_durations.append(round_duration)

        # logging.info(f"Selected participants to run: {clientsToRun}:\n{virtual_client_clock}")
        logging.info(f"Selected participants to run: {clientsToRun}\n")

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(clientsToRun)
        self.tasks_round = len(clientsToRun)

        self.save_last_param()
        self.round_stragglers = round_stragglers
        self.virtual_client_clock = virtual_client_clock
        self.flatten_client_duration = numpy.array(flatten_client_duration)
        self.round_duration = round_duration
        self.model_in_update = 0
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []

        if self.epoch >= self.args.epochs:
            self.event_queue.append('stop')
        elif self.epoch % self.args.eval_interval == 0:
            self.event_queue.append('update_model')
            self.event_queue.append('update_mask')
            self.event_queue.append('test')
        else:
            self.event_queue.append('update_model')
            self.event_queue.append('update_mask')
            self.event_queue.append('start_round')


    def testing_completion_handler(self, responses):
        """Each executor will handle a subset of testing dataset
        """
        response = pickle.loads(responses.result().serialized_test_response)
        executorId, results = response['executorId'], response['results']

        # List append is thread-safe
        self.test_result_accumulator.append(results)

        # Have collected all testing results
        if len(self.test_result_accumulator) == len(self.executors):
            accumulator = self.test_result_accumulator[0]
            for i in range(1, len(self.test_result_accumulator)):
                if self.args.task == "detection":
                    for key in accumulator:
                        if key == "boxes":
                            for j in range(self.imdb.num_classes):
                                accumulator[key][j] = accumulator[key][j] + self.test_result_accumulator[i][key][j]
                        else:
                            accumulator[key] += self.test_result_accumulator[i][key]
                else:
                    for key in accumulator:
                        accumulator[key] += self.test_result_accumulator[i][key]
            if self.args.task == "detection":
                self.testing_history['perf'][self.epoch] = {'round': self.epoch, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']*100.0/len(self.test_result_accumulator), 4),
                    'top_5': round(accumulator['top_5']*100.0/len(self.test_result_accumulator), 4),
                    'loss': accumulator['test_loss'],
                    'test_len': accumulator['test_len']
                    }
            else:
                self.testing_history['perf'][self.epoch] = {'round': self.epoch, 'clock': self.global_virtual_clock,
                    'top_1': round(accumulator['top_1']/accumulator['test_len']*100.0, 4),
                    'top_5': round(accumulator['top_5']/accumulator['test_len']*100.0, 4),
                    'loss': accumulator['test_loss']/accumulator['test_len'],
                    'test_len': accumulator['test_len']
                    }


            logging.info("FL Testing in epoch: {}, virtual_clock: {}, top_1: {} %, top_5: {} %, test loss: {:.4f}, test len: {}"
                    .format(self.epoch, self.global_virtual_clock, self.testing_history['perf'][self.epoch]['top_1'],
                    self.testing_history['perf'][self.epoch]['top_5'], self.testing_history['perf'][self.epoch]['loss'],
                    self.testing_history['perf'][self.epoch]['test_len']))

            # Dump the testing result
            with open(os.path.join(logDir, 'testing_perf'), 'wb') as fout:
                pickle.dump(self.testing_history, fout)

            if len(self.loss_accumulator):
                self.log_writer.add_scalar('Test/round_to_loss', self.testing_history['perf'][self.epoch]['loss'], self.epoch)
                self.log_writer.add_scalar('Test/round_to_accuracy', self.testing_history['perf'][self.epoch]['top_1'], self.epoch)
                self.log_writer.add_scalar('FAR/time_to_test_loss (min)', self.testing_history['perf'][self.epoch]['loss'],
                                            self.global_virtual_clock/60.)
                self.log_writer.add_scalar('FAR/time_to_test_accuracy (min)', self.testing_history['perf'][self.epoch]['top_1'],
                                            self.global_virtual_clock/60.)

            self.event_queue.append('start_round')


    def get_client_conf(self, clientId):
        # learning rate scheduler
        conf = {}
        conf['learning_rate'] = self.args.learning_rate
        conf['model'] = None
        return conf

    def create_client_task(self, executorId):
        """Issue a new client training task to the executor"""

        next_clientId = self.resource_manager.get_next_task()

        if next_clientId is not None:
            config = self.get_client_conf(next_clientId)

            future_response = self.executors.get_stub(executorId).Train.future(
                job_api_pb2.TrainRequest(client_id=next_clientId, serialized_train_config=pickle.dumps(config)))

            future_response.add_done_callback(self.task_completion_handler)


    def fetch_completion_handler(self, responses):
        training_result = pickle.loads(responses.result().serialized_fetch_response)
        self.client_result_queue.append(training_result)


    def task_completion_handler(self, responses):
        """Handler for training completion on each executor"""

        response = pickle.loads(responses.result().serialized_train_response)
        executorId, results = response.executorId, response.status

        # Schedule a new task first to pipeline computation and communication
        self.create_client_task(executorId)

        # Fetch model updates
        if results is False:
            logging.error(f"Executor {executorId} fails to run client {response.clientId}, due to {response.error}")

        fetch_response = self.executors.get_stub(executorId).Fetch.future(
                                job_api_pb2.FetchRequest(client_id=response.clientId))

        fetch_response.add_done_callback(self.fetch_completion_handler)


    def event_monitor(self):
        logging.info("Start monitoring events ...")
        start_time = time.time()
        time.sleep(10)

        while time.time() - start_time < 2000:
            try:
                self.executors.open_grpc_connection()
                for executorId in self.executors:
                    response = self.executors.get_stub(executorId).ReportExecutorInfo(
                        job_api_pb2.ReportExecutorInfoRequest())
                    self.executor_info_handler(executorId, {"size": response.training_set_size})
                break

            except Exception as e:
                self.executors.close_grpc_connection()
                logging.warning(f"{e}: Have not received executor information. This may due to slow data loading (e.g., Reddit) {traceback.format_exc()}")
                time.sleep(30)

        logging.info("Have received all executor information")

        while True:
            if len(self.event_queue) != 0:
                event_msg = self.event_queue.popleft()

                if event_msg == 'update_model':
                    serialized_data = pickle.dumps(self.model.to(device='cpu'))

                    future_context = []
                    for executorId in self.executors:
                        future_context.append(self.executors.get_stub(executorId).UpdateModel.future(
                            job_api_pb2.UpdateModelRequest(serialized_tensor=serialized_data)))

                    for context in future_context:
                        _ = context.result()

                elif event_msg == "update_mask":
                    serialized_data = pickle.dumps(self.mask_model)

                    future_context = []
                    for executorId in self.executors:
                        future_context.append(self.executors.get_stub(executorId).UpdateModel.future(
                            job_api_pb2.UpdateModelRequest(serialized_tensor=serialized_data)))

                    for context in future_context:
                        _ = context.result()
                    
                elif event_msg == 'start_round':
                    for executorId in self.executors:
                        self.create_client_task(executorId)

                elif event_msg == 'stop':
                    for executorId in self.executors:
                        _ = self.executors.get_stub(executorId).Stop.future(job_api_pb2.StopRequest())

                    self.stop()
                    break

                elif event_msg == 'test':
                    for executorId in self.executors:
                        future_response = self.executors.get_stub(executorId).Test.future(job_api_pb2.TestRequest())
                        future_response.add_done_callback(self.testing_completion_handler)

            elif len(self.client_result_queue) > 0:
                self.client_completion_handler(self.client_result_queue.pop(0))
                if len(self.stats_util_accumulator) == self.tasks_round:
                        self.round_completion_handler()
            else:
                # execute every 100 ms
                time.sleep(0.1)

        self.executors.close_grpc_connection()


    def stop(self):
        logging.info(f"Terminating the aggregator ...")
        time.sleep(5)

if __name__ == "__main__":
    aggregator = Aggregator(args)
    aggregator.run()