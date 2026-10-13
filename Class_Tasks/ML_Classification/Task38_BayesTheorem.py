# Practice Q1 - Change P(free|ham) from 0.05 to 0.30. What happens to P(spam|free)? Why?
p_spam = 0.30
p_ham = 1 - p_spam
p_free_given_spam = 0.60

for p_free_given_ham in [0.05, 0.30]:
    p_free = p_free_given_spam * p_spam + p_free_given_ham * p_ham
    p_spam_given_free = p_free_given_spam * p_spam / p_free
    print(f"P(free|ham) = {p_free_given_ham}: P(spam|free) = {round(p_spam_given_free, 3)}")

# P(spam|free) drops from 0.84 to 0.46. When the word "free" shows up a lot
# in normal emails too, seeing it is much weaker evidence for spam.
