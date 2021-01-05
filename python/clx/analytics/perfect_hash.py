import numpy as np
import argparse

np.random.seed(1243342)

PRIME = np.uint64(281474976710677)

# Coefficients ranges for inner hash - This are important to set to be
# large so that we have randomness in the bottom bits when modding
A_SECOND_LEVEL_POW = np.uint8(48)
B_SECOND_LEVEL_POW = np.uint8(7)

A_LBOUND_SECOND_LEVEL_HASH = 2**16
A_HBOUND_SECOND_LEVEL_HASH = 2**A_SECOND_LEVEL_POW

B_LBOUND_SECOND_LEVEL_HASH = 0
B_HBOUND_SECOND_LEVEL_HASH = 2**B_SECOND_LEVEL_POW

# Extremely generous and should not ever happen. This limit is imposed
# To ensure we can bit pack all the information needed for the bin hash
# functions - a, b and table size
MAX_SIZE_FOR_INITIAL_BIN = 2**8 - 1


#Shifts for bit packing
A_SECOND_LEVEL_SHIFT_AMT = np.uint8(64 - A_SECOND_LEVEL_POW)
B_SECOND_LEVEL_SHIFT_AMT = np.uint8(64 - A_SECOND_LEVEL_POW - B_SECOND_LEVEL_POW)
BITS_FOR_INNER_TABLE_SIZE = np.uint8(8)

NOT_FOUND = -1

def sdbm_hash(string):
  hv = 0
  mask = (1 << 48) - 1
  for c in string:
      hv = ord(c) + (hv << 6) + (hv << 16) - hv
      hv &= mask
  return hv 


def hash_func(k, a, b, size):
  k = np.uint64(k)
  a = np.uint64(a)
  b = np.uint64(b)
  size = np.uint64(size)
  return ((a*k + b) % PRIME) % size


def longest_bin_length(bins):
  return len(max(bins, key=len))
  
  
def make_bins(data, num_bins, a, b):
  h = lambda k: hash_func(k, a, b, num_bins)

  bins = [[] for i in range(num_bins)]

  for item in data:
      bins[h(item)].append(item)
  return bins


def new_bin_length(orig_length, compact):
  return int(orig_length) if compact else orig_length**2


def get_space_util(bins, init_bins, compact):
  return sum(new_bin_length(len(b), compact) for b in bins) + 2*init_bins


def pick_initial_a_b(data, max_constant, init_bins, compact):
  while True:
    a = np.random.randint(2**12, 2 ** 15)
    b = np.random.randint(2**12, 2 ** 15)
    bins = make_bins(data,  init_bins, a, b)
    score = get_space_util(bins, init_bins, compact) / len(data)

    longest = new_bin_length(longest_bin_length(bins), compact)

    if score <= max_constant and longest <= MAX_SIZE_FOR_INITIAL_BIN:
      print("Attempting to build table using {:.6f}n space".format(score))
      print("Longest bin was {}".format(longest))
      break

  return bins, a, b


def find_hash_for_internal(hash_bin, compact):
  if not hash_bin:
    return [[], 0, 0]
  
  new_length = new_bin_length(len(hash_bin), compact)

  while True:
    a = np.random.randint(A_LBOUND_SECOND_LEVEL_HASH, A_HBOUND_SECOND_LEVEL_HASH)
    b = np.random.randint(B_LBOUND_SECOND_LEVEL_HASH, B_HBOUND_SECOND_LEVEL_HASH)
    bins = make_bins(hash_bin, new_length, a, b)

    max_length = len(max(bins, key=len))
    if max_length == 1:
      bins = [b[0] if b else 0 for b in bins]
      return bins, a, b

def perfect_hash(integers, max_constant, compact):
  num_top_level_bins = len(integers)//4

  init_bins, init_a, init_b = pick_initial_a_b(integers, max_constant, num_top_level_bins, compact)
  flattened_bins = []
  
  internal_table_coeffs = np.zeros(shape=[num_top_level_bins], dtype=np.uint64)
  offset_into_flattened_table = np.zeros(shape=[num_top_level_bins + 1], dtype=np.uint64)

  max_bin_length = 0
  for i, b in enumerate(init_bins):
    print("Processing bin", i, "size", len(b))
    internal_table, coeff_a, coeff_b = find_hash_for_internal(b, compact)
    bin_length = len(internal_table)
    max_bin_length = max(bin_length, max_bin_length)
    internal_table_coeffs[i] = coeff_a << A_SECOND_LEVEL_SHIFT_AMT | coeff_b << B_SECOND_LEVEL_SHIFT_AMT | bin_length
    offset_into_flattened_table[i + 1] = offset_into_flattened_table[i] + bin_length
    flattened_bins.extend(internal_table)

  print("Final table size {} elements compared to {} for original".
        format(len(flattened_bins), len(integers)))
  
  print("Max bin length was", max_bin_length)

  return init_a, init_b, num_top_level_bins, flattened_bins, internal_table_coeffs, offset_into_flattened_table


def pack_keys_and_values(flattened_hash_table, original_dict):

  for i in range(len(flattened_hash_table)):
    if flattened_hash_table[i] in original_dict:
      value = original_dict[flattened_hash_table[i]]
      flattened_hash_table[i] <<= 16
      flattened_hash_table[i] |= value 


def load_vocab_dict(path):
  vocab = {}
  with open(path, mode="r") as f:
    counter = 0
    for line in f:
      vocab[line.strip()] = counter
      counter += 1
  
  return vocab


def hash_vocab(path, store_path, compact, unk_tok="[UNK]", first_token="[CLS]", sep_token="[SEP]"):
  vocab = load_vocab_dict(path)
  keys = list(map(sdbm_hash, vocab.keys()))
  
  hashed_vocab = {sdbm_hash(key) : value for key, value in vocab.items()}

  assert len(hashed_vocab) == len(vocab), "Collision occurred and only sdbm token hash current supported :(. \
                                           Can be extended to use random hashes if needed"
  
  outer_a, outer_b, num_outer_bins, hash_table, inner_table_coeffs, offsets_into_ht = perfect_hash(keys, 10, compact)

  pack_keys_and_values(hash_table, hashed_vocab)
  store_func(store_path, outer_a, outer_b, num_outer_bins, hash_table, inner_table_coeffs, offsets_into_ht, 
             vocab[unk_tok], vocab[first_token], vocab[sep_token])

  for key, value in hashed_vocab.items():
    val = retrieve(key, outer_a, outer_b, num_outer_bins, hash_table, inner_table_coeffs, offsets_into_ht)
    assert val == value, "Incorrect value found. Got {} expected {}".format(val, value)
  
  print("All present tokens return correct value.")


def store_func(out_name, outer_a, outer_b, num_outer_bins, hash_table, inner_table_coeffs, offsets_into_ht, 
               unk_tok_id, first_token_id, sep_token_id):

  with open(out_name, mode="w+") as f:
    f.write("{}\n".format(outer_a))
    f.write("{}\n".format(outer_b))
    f.write("{}\n".format(num_outer_bins))
    f.writelines("{} {}\n".format(coeff, offset) for coeff, offset in zip(inner_table_coeffs, offsets_into_ht))
    f.write("{}\n".format(len(hash_table)))
    f.writelines("{}\n".format(kv) for kv in hash_table)
    f.writelines("{}\n".format(tok_id) for tok_id in [unk_tok_id, first_token_id, sep_token_id])


def retrieve(k, outer_a, outer_b, num_outer_bins, hash_table, inner_table_coeffs, offsets_into_ht):

  bin_hash = hash_func(k, outer_a, outer_b, num_outer_bins)
  start_offset_in_ht = offsets_into_ht[bin_hash]
  inner_table_values = inner_table_coeffs[bin_hash]

  one = np.uint64(1)

  inner_a = inner_table_values >> A_SECOND_LEVEL_SHIFT_AMT
  inner_b = (inner_table_values >> B_SECOND_LEVEL_SHIFT_AMT) & ((one << B_SECOND_LEVEL_POW) - one) 
  size = inner_table_values & ((one << BITS_FOR_INNER_TABLE_SIZE) - one)

  inner_offset = hash_func(k, inner_a, inner_b, size)
  kv = hash_table[start_offset_in_ht + inner_offset]

  key, value = kv >> 16, kv & ((1 << 16) - 1)
  indicator = key == k
  
  return indicator*value + (not indicator)*NOT_FOUND

def sdbm_pop(h, last_val):
  mod_inverse = 24320495251391
  mask = (1 << 48) - 1
  return ( ((mod_inverse * h) & mask) - ((mod_inverse * last_val) & mask) & mask )  

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Construct a perfect hash table for the given vocabulary file")

  parser.add_argument("--vocab", "-v", required=True, dest="vocab", type=str, help="The path to the bert vocabulary file. Normally called vocab.txt")
  parser.add_argument("--output", "-o", required=True, dest="output", type=str, help="The location to store the output")
  parser.add_argument("--compact", action="store_true", dest="compact", help="If set, minimizes space at the expense of longer preprocessing.")
  parser.set_defaults(compact=False)
  ns = parser.parse_args()

  hash_vocab(ns.vocab, ns.output, ns.compact)