def ryerson_letter_grade(n):
    if n < 50:
        return 'F'
    elif n > 89:
        return 'A+'
    elif n > 84:
        return 'A'
    elif n > 79:
        return 'A-'
    tens = n // 10
    ones = n % 10
    if ones < 3:
        adjust = "-"
    elif ones > 6:
        adjust = "+"
    else:
        adjust = ""
    return "DCB"[tens - 5] + adjust


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def all_factors(n):
    factors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
        i += 1
    return sorted(factors)


def fibonacci_sum(n):
    if n < 1:
        return []
    fibs = [1, 2]
    while fibs[-1] < n:
        fibs.append(fibs[-1] + fibs[-2])
    result = []
    for fib in reversed(fibs):
        if fib <= n:
            result.append(fib)
            n -= fib
    return result


def reverse_vowels(text):
    vowels = 'aeiouAEIOU'
    text_list = list(text)
    vowel_positions = [i for i, c in enumerate(text_list) if c in vowels]
    vowel_chars = [text_list[i].lower() for i in vowel_positions]
    vowel_chars.reverse()
    for i, pos in enumerate(vowel_positions):
        original_char = text_list[pos]
        new_vowel = vowel_chars[i]
        if original_char.isupper():
            new_vowel = new_vowel.upper()
        text_list[pos] = new_vowel
    return ''.join(text_list)


def remove_after_kth(items, k=1):
    if k < 1:
        return []
    counts = {}
    result = []
    for item in items:
        count = counts.get(item, 0)
        if count < k:
            result.append(item)
            counts[item] = count + 1
    return result


def goldbach(n):
    for p in range(2, n):
        if is_prime(p) and is_prime(n - p):
            return (p, n - p)
    return None


def sum_of_two_squares(n):
    i = 1
    while i * i <= n:
        j_squared = n - i * i
        j = int(j_squared ** 0.5)
        if j * j == j_squared and j > 0:
            return (max(i, j), min(i, j))
        i += 1
    return None


def count_carries(a, b):
    carry = 0
    count = 0
    while a > 0 or b > 0:
        digit_sum = (a % 10) + (b % 10) + carry
        if digit_sum >= 10:
            count += 1
            carry = 1
        else:
            carry = 0
        a //= 10
        b //= 10
    return count


def first_preceded_by_smaller(items, k=1):
    for i in range(len(items)):
        count_smaller = sum(1 for j in range(i) if items[j] < items[i])
        if count_smaller >= k:
            return items[i]
    return None


def count_divisibles_in_range(start, end, n):
    if n == 0:
        return 0
    first = start + (n - start % n) % n
    if first > end:
        return 0
    return (end - first) // n + 1