import math


prob = 0.95

res = prob ** 100 - prob ** 89

# prev = prob ** 90
# res = prev

# for i in range(90, 101):
#     prev = prev * prob
#     res += prev

# print(res)

res = 0
for i in range(90, 101):
    res += math.comb(100, i) * (prob ** i) / math.factorial(100)

print(res)


# print(math.comb(100, 90) / math.factorial(100))
