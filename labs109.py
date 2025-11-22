import math
import re
import calendar
import datetime
from functools import reduce, cmp_to_key, lru_cache
from itertools import combinations, chain, islice, count, product, zip_longest, cycle
from math import isqrt, gcd
from decimal import Decimal, getcontext
from collections import Counter, deque
from heapq import heappush, heappop, heapify
from datetime import timedelta
from fractions import Fraction
from queue import Queue
from bisect import bisect_left


"""
NOTES:

Problem Set A - 109 Python Problems for CCPS 109.pdf
Problem Set B - Additional Python Problems.pdf
Problem Set C - Third Python Problem Collection.pdf
"""

# Problem A1: Ryerson letter grade
def ryerson_letter_grade(pct):
    if pct < 50:
        return 'F'
    elif pct > 89:
        return 'A+'
    elif pct > 84:
        return 'A'
    elif pct > 79:
        return 'A-'
    tens = pct // 10
    ones = pct % 10
    if ones < 3:
        adjust = "-"
    elif ones > 6:
        adjust = "+"
    else:
        adjust = ""
    return "DCB"[tens - 5] + adjust


# Problem A2: Ascending list
def is_ascending(items):
    if len(items) <= 1:
        return True

    for i in range(1, len(items)):
        if items[i] <= items[i-1]:
            return False
    return True


# Problem A3: Riffle shuffle kerfuffle
def riffle(items, out=True):
    n = len(items)
    half = n // 2

    if n == 0:
        return []

    first_half = items[:half]
    second_half = items[half:]
    result = []

    if out:
        first, second = first_half, second_half
    else:
        first, second = second_half, first_half

    for i in range(half):
        result.append(first[i])
        result.append(second[i])

    return result


# Problem A4: Even the odds
def only_odd_digits(n):
    if n <= 0:
        return False 

    s_n = str(n)
    for digit in s_n:
        if int(digit) % 2 == 0:
            return False
    return True


# Problem A5: Cyclops numbers
def is_cyclops(n):
    s_n = str(n)
    length = len(s_n)

    if length % 2 == 0:
        return False

    if n == 0:
        return True

    middle_index = length // 2

    if s_n[middle_index] != '0':
        return False

    for i, digit_char in enumerate(s_n):
        if i != middle_index:
            if digit_char == '0':
                return False

    return True


# Problem A6: Domino cycle
def domino_cycle(tiles):
    n = len(tiles)

    for i in range(n):
        current_end = tiles[i][1]

        successor_start = tiles[(i + 1) % n][0]

        if current_end != successor_start:
            return False

    return True


# Problem A7: Colour trio
def colour_trio(colours):

    mix = {
        ('r', 'r'): 'r', ('y', 'y'): 'y', ('b', 'b'): 'b',
        ('r', 'y'): 'b', ('y', 'r'): 'b',
        ('r', 'b'): 'y', ('b', 'r'): 'y',
        ('y', 'b'): 'r', ('b', 'y'): 'r'
    }

    current_row = colours

    while len(current_row) > 1:
        next_row = ""
        for i in range(len(current_row) - 1):
            color1 = current_row[i]
            color2 = current_row[i+1]
            next_row += mix[(color1, color2)]

        current_row = next_row

    return current_row[0]


# Problem A8: Count dominators
def count_dominators(items):
    n = len(items)
    if n == 0:
        return 0

    dominator_count = 1
    max_to_the_right = items[n - 1]

    for i in range(n - 2, -1, -1):
        current_item = items[i]

        if current_item > max_to_the_right:
            dominator_count += 1
            max_to_the_right = current_item

    return dominator_count


# Problem A9: Beat the previous
def extract_increasing(digits):
    result = []
    n = len(digits)
    current_index = 0
    prev_num = -1 

    while current_index < n:
        length = 1
        found_next = False

        while current_index + length <= n:
            current_substring = digits[current_index : current_index + length]
            current_num = int(current_substring)

            if current_num > prev_num:
                result.append(current_num)
                prev_num = current_num
                current_index += length
                found_next = True
                break

            length += 1

        if not found_next:
            break

    return result


# Problem A11: Taxi Zum Zum
def taxi_zum_zum(moves):
    x, y = 0, 0
    dx, dy = 0, 1

    for move_char in moves:
        char = move_char.upper()

        if char == 'F':
            x += dx
            y += dy

        elif char == 'R':
            dx, dy = dy, -dx

        elif char == 'L':
            dx, dy = -dy, dx

    return (x, y)


# Problem A13: Rook on a rampage
def safe_squares_rooks(n, rooks):

    occupied_rows = set()
    occupied_cols = set()

    for r, c in rooks:
        occupied_rows.add(r)
        occupied_cols.add(c)

    num_threatened_rows = len(occupied_rows)
    num_threatened_cols = len(occupied_cols)
    num_safe_rows = n - num_threatened_rows
    num_safe_cols = n - num_threatened_cols

    return num_safe_rows * num_safe_cols


# Problem A16: The card that wins the trick
def winning_card(cards, trump=None):
    RANK_VALUES = {
        'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 
        'eight': 8, 'nine': 9, 'ten': 10, 'jack': 11, 'queen': 12, 
        'king': 13, 'ace': 14
    }

    lead_card_tuple = cards[0]
    lead_suit = lead_card_tuple[1]
    winning_card_tuple = lead_card_tuple

    def get_rank_value(card_tuple):
        rank_str = card_tuple[0]

        return RANK_VALUES[rank_str.lower()]

    def compare_cards(card1_tuple, card2_tuple, lead_suit, trump_suit):
        rank1, suit1 = card1_tuple[0], card1_tuple[1]
        rank2, suit2 = card2_tuple[0], card2_tuple[1]
        value1 = get_rank_value(card1_tuple)
        value2 = get_rank_value(card2_tuple)
        is1_trump = (suit1 == trump_suit)
        is2_trump = (suit2 == trump_suit)
        is1_lead = (suit1 == lead_suit)
        is2_lead = (suit2 == lead_suit)

        if is1_trump and not is2_trump:
            return card1_tuple
        if is2_trump and not is1_trump:
            return card2_tuple
        if is1_trump and is2_trump:
            return card1_tuple if value1 > value2 else card2_tuple

        if is1_lead and not is2_lead:
            return card1_tuple
        if is2_lead and not is1_lead:
            return card2_tuple
        if is1_lead and is2_lead:
            return card1_tuple if value1 > value2 else card2_tuple

    for current_card_tuple in cards[1:]:
        winning_card_tuple = compare_cards(winning_card_tuple, current_card_tuple, lead_suit, trump)

    return winning_card_tuple


# Problem A20: Fail while daring greatly
def josephus(n, k):

    if n <= 0:
        return []

    people = list(range(1, n + 1))
    elimination_sequence = []
    current_index = 0

    while len(people) > 0:

        if len(people) > 1:
            current_index = (current_index + (k - 1)) % len(people)
        else:
            current_index = 0

        eliminated = people.pop(current_index)
        elimination_sequence.append(eliminated)

    return elimination_sequence


# Problem A29: Between the soft and the NP-hard place
def verify_betweenness(perm, constraints):
    index_map = {element: i for i, element in enumerate(perm)}

    for a, b, c in constraints:
        try:
            index_a = index_map[a]
            index_b = index_map[b]
            index_c = index_map[c]
        except KeyError:
            return False

        is_satisfied = (index_a < index_b < index_c) or (index_c < index_b < index_a)

        if not is_satisfied:
            return False

    return True


# Problem A32: Three summers ago
def three_summers(items, goal):

    items.sort()
    n = len(items)

    for i in range(n - 2):
        a = items[i]

        if i > 0 and a == items[i - 1]:
            continue

        left = i + 1
        right = n - 1
        target_sum = goal - a

        while left < right:
            current_sum = items[left] + items[right]

            if current_sum == target_sum:
                return True

            elif current_sum < target_sum:
                left += 1
            else: 
                right -= 1

    return False


# Problem A33: Sum of two Squares
def sum_of_two_squares(n):
    i = 1
    while i * i <= n:
        j_squared = n - i * i
        j = int(j_squared ** 0.5)
        if j * j == j_squared and j > 0:
            return (max(i, j), min(i, j))
        i += 1
    return None


# Problem A34: Carry on Pythonista
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


# Problem A36: Expand positive integer intervals
def expand_intervals(intervals):
    if isinstance(intervals, str):
        parsed_intervals = []
        for interval_str in intervals.split(','):
            interval_str = interval_str.strip()
            if not interval_str:
                continue

            if '-' in interval_str:
                try:
                    start_str, end_str = interval_str.split('-', 1)
                    start = int(start_str)
                    end = int(end_str)
                    parsed_intervals.append([start, end])
                except ValueError:
                    continue
            else:
                try:
                    num = int(interval_str)
                    parsed_intervals.append([num, num])
                except ValueError:
                    continue
        intervals = parsed_intervals

    all_numbers = set()

    for start, end in intervals:
        if start <= end:
            for num in range(start, end + 1):
                all_numbers.add(num)

    return sorted(list(all_numbers))


# Problem A37: Collapse positive integer intervals
def collapse_intervals(items):

    if not items:
        return ""

    sorted_unique_items = sorted(list(set(items)))

    if not sorted_unique_items:
        return "" 

    collapsed_intervals = []
    start = sorted_unique_items[0]

    for i in range(1, len(sorted_unique_items)):
        current_num = sorted_unique_items[i]
        previous_num = sorted_unique_items[i - 1]

        if current_num > previous_num + 1:
            end = previous_num
            collapsed_intervals.append([start, end])
            start = current_num

    collapsed_intervals.append([start, sorted_unique_items[-1]])

    if len(collapsed_intervals) == 1:
        start, end = collapsed_intervals[0]
        if start == end:
            return start
        else:
            return f"{start}-{end}"

    output_parts = []
    for start, end in collapsed_intervals:
        if start == end:
            output_parts.append(str(start))
        else:
            output_parts.append(f"{start}-{end}")

    return ",".join(output_parts)


# Problem A43: Interesting, intersecting
def squares_intersect(s1, s2):

    x1_min = s1[0]
    y1_min = s1[1]
    s1_len = s1[2]
    x1_max = x1_min + s1_len
    y1_max = y1_min + s1_len

    x2_min = s2[0]
    y2_min = s2[1]
    s2_len = s2[2]
    x2_max = x2_min + s2_len
    y2_max = y2_min + s2_len

    no_x_overlap = (x1_max < x2_min) or (x2_max < x1_min)
    no_y_overlap = (y1_max < y2_min) or (y2_max < y1_min)
    do_not_intersect = no_x_overlap or no_y_overlap

    return not do_not_intersect


# Problem A45: That's enough of you!
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


# Problem A49: That's Enough for you!
def first_preceded_by_smaller(items, k=1):
    for i in range(len(items)):
        count_smaller = sum(1 for j in range(i) if items[j] < items[i])
        if count_smaller >= k:
            return items[i]
    return None


# Problem A51: What do you hear, what do you say?
def count_and_say(digits):

    current_sequence = str(digits)

    if not current_sequence:
        return ""

    result = []
    i = 0
    n = len(current_sequence)

    while i < n:
        current_digit = current_sequence[i]
        count = 1
        j = i + 1

        while j < n and current_sequence[j] == current_digit:
            count += 1
            j += 1

        result.append(str(count))
        result.append(current_digit)
        i = j

    return "".join(result)


# Problem A55: Revorse the vewels
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


# Problem A78: Count divisibles in range
def count_divisibles_in_range(start, end, n):
    if n == 0:
        return 0

    first = start + (n - start % n) % n

    if first > end:
        return 0
    return (end - first) // n + 1


# Problem A86: Calling all units, B-and-E in progress
def is_perfect_power(n):
    if n <= 3:
        return n == 4 # 4 = 2^2

    max_exponent = n.bit_length()

    for m in range(2, max_exponent):
        root_float = n ** (1.0 / m)
        k1 = int(round(root_float))
        if k1 >= 2 and pow(k1, m) == n:
            return True

    return False


# Problem A89: Fibonacci sum
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


# Problem B1: The Fischer King 
def is_chess_960(row):

    bishop_indices = [i for i, piece in enumerate(row) if piece == 'b']
    if len(bishop_indices) != 2 or bishop_indices[0] % 2 == bishop_indices[1] % 2:
        return False

    rook_indices = [i for i, piece in enumerate(row) if piece == 'r']
    king_index = row.find('K')

    if len(rook_indices) != 2:
        return False

    min_rook_index = min(rook_indices)
    max_rook_index = max(rook_indices)

    if not (min_rook_index < king_index < max_rook_index):
        return False

    return True


# Problem B2: Multiplicative persistence
def multiplicative_persistence(n, ignore_zeros=False):

    n_str = str(n)
    if len(n_str) <= 1:
        return 0

    persistence = 0
    current_n = n

    def next_value(num_str, ignore_zeros):
        product = 1
        has_non_zero = False
        for digit_char in num_str:
            digit = int(digit_char)
            if digit == 0 and not ignore_zeros:
                return 0
            if digit > 0:
                product *= digit
                has_non_zero = True

        if ignore_zeros and not has_non_zero and '0' in num_str:
            return 0

        return product

    while len(str(current_n)) > 1:
        current_n = next_value(str(current_n), ignore_zeros)
        persistence += 1

    return persistence


# Problem B3: Top of the swops
def topswops(cards):

    cards_list = list(cards)
    moves = 0

    while cards_list[0] != 1:
        k = cards_list[0]
        cards_list[:k] = cards_list[:k][::-1]
        moves += 1

    return moves


# Problem B5: Discrete rounding
def discrete_rounding(n):

    current_num = n
    for k in range(n - 1, 1, -1):
        if current_num % k == 0:
            continue
        else:
            current_num = current_num + (k - (current_num % k))

    return current_num


# Problem B6: Translate
def tr(text, ch_from, ch_to):
    mapping = str.maketrans(ch_from, ch_to)
    return text.translate(mapping)


# Problem B7: Ifs and butts 
def count_cigarettes(n, k):

    total_smoked = n
    butts = n 

    while butts >= k:
        new_cigarettes = butts // k

        total_smoked += new_cigarettes
        butts = (butts % k) + new_cigarettes

    return total_smoked


# Problem B8: Word positions
def word_positions(sentence, word):

    words_list = sentence.split()

    positions = []
    for i, current_word in enumerate(words_list):
        if current_word == word:
            positions.append(i)

    return positions


# Problem B10: Deterministic finite automata 
def dfa(rules, text):

    current_state = 0

    for char in text:
        key = (current_state, char)
        if key in rules:
            current_state = rules[key]
        else:
            break

    return current_state


# Problem B12: Lychrel numbers
def lychrel(n, giveup):

    def is_palindrome(num):
        s = str(num)
        return s == s[::-1]

    def reverse_and_add(num):
        s = str(num)
        reversed_s = s[::-1]
        return num + int(reversed_s)

    current_number = n

    if is_palindrome(current_number):
        return 0

    for steps in range(1, giveup + 1):
        current_number = reverse_and_add(current_number)

        if is_palindrome(current_number):
            return steps

    return None


# Problem B15: Count possible triangles
def count_triangles(sides):

    n = len(sides)
    count = 0
    sides.sort()

    for k in range(n):
        c = sides[k]
        i = 0
        j = k - 1

        while i < j:
            a = sides[i]
            b = sides[j]

            if a + b > c:
                count += (j - i)
                j -= 1
            else:
                i += 1

    return count


# Problem B17: Count Friday the Thirteenths
def count_friday_13s(start, end):

    is_date_input = isinstance(start, datetime.date)

    if is_date_input:
        start_year = start.year
        start_month = start.month
        end_year = end.year
        end_month = end.month
    else:
        start_year = start
        start_month = 1
        end_year = end
        end_month = 12

    total_friday_13s = 0

    for year in range(start_year, end_year + 1):

        current_start_month = start_month if year == start_year else 1
        current_end_month = end_month if year == end_year else 12

        for month in range(current_start_month, current_end_month + 1):

            day_to_check = 13

            if is_date_input:
                if year == start_year and month == start_month and day_to_check < start.day:
                    continue
                if year == end_year and month == end_month and day_to_check > end.day:
                    continue

            d = day_to_check
            Y = year
            m = month

            if m <= 2:
                m += 12
                Y -= 1

            K = Y % 100 
            J = Y // 100 
            h = (d + (13 * (m + 1) // 5) + K + (K // 4) + (J // 4) - (2 * J)) % 7

            if h == 6:
                total_friday_13s += 1

    return total_friday_13s


# Problem B21: Nondeterministic finite automata
def nfa(rules, text):

    start_state=0
    current_states = {start_state}

    for char in text:
        next_states = set()
        for state in current_states:
            key = (state, char)
            successor_list = rules.get(key, [])
            next_states.update(successor_list)

        current_states = next_states

    return sorted(list(current_states))


# Problem B33: Ten pins, not six, Dolores
def bowling_score(frames):

    flat_rolls = []

    for i, frame_str in enumerate(frames):
        if i < 9: 
            if frame_str == 'X':
                flat_rolls.append(10)
            elif frame_str[1] == '/':
                roll1_score = int(frame_str[0]) if frame_str[0].isdigit() else 0
                flat_rolls.append(roll1_score)
                flat_rolls.append(10 - roll1_score)
            else: 
                roll1_score = int(frame_str[0]) if frame_str[0].isdigit() else 0
                roll2_score = int(frame_str[1]) if frame_str[1].isdigit() else 0
                flat_rolls.extend([roll1_score, roll2_score])

        else: 
            roll_scores = []

            if frame_str[0] == 'X':
                roll_scores.append(10)
            elif frame_str[0] == '-':
                roll_scores.append(0)
            else:
                roll_scores.append(int(frame_str[0]))

            if frame_str[1] == '/':
                roll_scores.append(10 - roll_scores[0])
            elif frame_str[1] == 'X':
                roll_scores.append(10)
            elif frame_str[1] == '-':
                roll_scores.append(0)
            else:
                roll_scores.append(int(frame_str[1]))

            if len(frame_str) == 3:
                if frame_str[2] == 'X':
                    roll_scores.append(10)
                elif frame_str[2] == '/':
                    roll_scores.append(10 - roll_scores[1])
                elif frame_str[2] == '-':
                    roll_scores.append(0)
                else:
                    roll_scores.append(int(frame_str[2]))

            flat_rolls.extend(roll_scores)

    score = 0
    roll_idx = 0

    for frame in range(10):
        if flat_rolls[roll_idx] == 10:
            score += 10 + flat_rolls[roll_idx + 1] + flat_rolls[roll_idx + 2]
            roll_idx += 1

        elif flat_rolls[roll_idx] + flat_rolls[roll_idx + 1] == 10:
            score += 10 + flat_rolls[roll_idx + 2]
            roll_idx += 2

        else:
            score += flat_rolls[roll_idx] + flat_rolls[roll_idx + 1]
            roll_idx += 2

    return score


# Problem B34: Longeset mirrored substring
def has_majority(items):

    n = len(items)
    if n == 0:
        return False

    candidate = None
    count = 0

    for item in items:
        if count == 0:
            candidate = item
            count = 1
        elif item == candidate:
            count += 1
        else:
            count -= 1

    actual_count = 0
    for item in items:
        if item == candidate:
            actual_count += 1

    return actual_count * 2 > n


# Problem B35: Add like an Egyptian
def greedy_egyptian(f):

    if isinstance(f, (list, tuple)) and len(f) == 2:
        a, b = f  
    elif isinstance(f, Fraction):
        a, b = f.numerator, f.denominator
    else:
        raise ValueError("Input must be a tuple/list (numerator, denominator) or a fractions.Fraction object.")

    f_current = Fraction(a, b)
    denominators = []

    while f_current.numerator > 0:
        a = f_current.numerator
        b = f_current.denominator

        x = (b + a - 1) // a
        denominators.append(x)
        unit_fraction = Fraction(1, x)
        f_current -= unit_fraction

    return denominators


# Problem B37: Van der Corput sequence
def van_der_corput(base, n):

    if base <= 1 or n < 0:
        raise ValueError("Base must be > 1 and n must be >= 0.")

    v_numerator = 0
    v_denominator = 1
    temp_n = n

    while temp_n > 0:
        v_denominator *= base
        digit = temp_n % base
        v_numerator = v_numerator * base + digit 
        temp_n //= base

    value = Fraction(0, 1)
    base_power = 1
    temp_n = n

    while temp_n > 0:
        digit = temp_n % base
        base_power *= base
        value += Fraction(digit, base_power)
        temp_n //= base

    return value


# Problem Set C (Page 14): Generalized Fibonacci sequence
def generalized_fibonacci(multipliers, n):

    m = len(multipliers)

    if n < m:
        return 1

    terms = [1] * m

    for i in range(m, n + 1):
        new_term = 0
        for j in range(m):
            new_term += multipliers[j] * terms[j]

        terms.pop(0)
        terms.append(new_term)

    return terms[-1]


# Problem Set C (Page 26): Primality buildup
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


# Problem Set C (Page 27): Goldbach verification
def goldbach(n):
    for p in range(2, n):
        if is_prime(p) and is_prime(n - p):
            return (p, n - p)
    return None


# Problem Set C (Page 30): Sum of square roots
def square_root_sum(n1, n2):

    def sum_sqrt(list):
        return sum(math.sqrt(x) for x in list)

    sum1 = sum_sqrt(n1)
    sum2 = sum_sqrt(n2)
    epsilon = 1e-9

    return sum1 >= sum2 - epsilon


# Problem Set C (Page 34): List of all factors
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


#Problem Set C (Page 45): Split at None
def split_at_none(items):
    result = []
    current_sublist = []

    for item in items:
        if item is None:
            result.append(current_sublist)
            current_sublist = []
        else:
            current_sublist.append(item)

    result.append(current_sublist)

    return result


# Problem Set C (Page 80): Expand recursively run-length encoded string
def expand_string(text):
    i = [0] 

    def decode_recursive():
        result = []
        num = 0
        N = len(text)

        while i[0] < N:
            char = text[i[0]]

            if '0' <= char <= '9':
                num = num * 10 + int(char)
                i[0] += 1

            elif char == '[':
                i[0] += 1
                sub_string = decode_recursive()
                K = num if num > 0 else 1 
                result.append(sub_string * K)
                num = 0 

            elif char == ']':
                i[0] += 1
                return "".join(result)

            else:
                K = num if num > 0 else 1
                result.append(char * K)
                i[0] += 1

                num = 0 

        return "".join(result)

    return decode_recursive()


# Problem Set C (Page 89): The prodigal sequence
def front_back_sort(perm):
    if len(perm) <= 1:
        return 0

    n = len(perm)
    pos = [0] * n

    for i, val in enumerate(perm):
        pos[val] = i

    max_length = 1
    current_length = 1

    for i in range(1, n):
        if pos[i] > pos[i-1]:
            current_length += 1
            max_length = max(max_length, current_length)
        else:
            current_length = 1

    return n - max_length
