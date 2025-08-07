paragraph = input("Enter a paragraph:\n")
import re
sentences = re.split(r'[.!?]+', paragraph)

sentences = [s for s in sentences if s.strip()]
num_sentences = len(sentences)

words = paragraph.split()
num_words = len(words)

vowels = 'aeiou'
vowel_freq = dict.fromkeys(vowels, 0)
for char in paragraph.lower():
    if char in vowels:
        vowel_freq[char] += 1


print("Number of sentences:", num_sentences)
print("Number of words:", num_words)
print("Frequency of each vowel:")
for v in vowels:
    print(f"{v}: {vowel_freq[v]}")