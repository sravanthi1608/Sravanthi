from math import comb

def binomial_probability(n, k, p):
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

p_right_handed = 0.9
sample_size = 10
total_probability = 0

for k in range(7):
    probability = binomial_probability(sample_size, k, p_right_handed)
    total_probability += probability

print("Probability of having at most 6 right-handed individuals:", total_probability)

