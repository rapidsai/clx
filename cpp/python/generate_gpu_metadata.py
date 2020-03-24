import unicodedata
import argparse
from itertools import groupby
from operator import itemgetter
import struct

SHIFT_FOR_NEW_CP = 0
SHIFT_FOR_BYTES_LESS_1 = 21
SHIFT_FOR_MULTICHAR = 23
SHIFT_FOR_TOK_CAT = 24

MAX_CODE_POINT = 1114111

REPLACEMENT_BYTES_MASK = 0xf
NEW_CP_MASK = (1 << 21) - 1
BYTES_LESS_1_MASK = 0x3
MULTI_CHAR_MASK = 1
TOKEN_CAT_MASK = 0x7
TOKEN_CAT_ADD_SPACE = 0
TOKEN_CAT_ADD_SPACE_IF_LOWER = 1
TOKEN_CAT_REMOVE_CHAR = 2
TOKEN_CAT_REMOVE_CHAR_IF_LOWER = 3
TOKEN_CAT_ALWAYS_REPLACE = 4


def gen_metadata(unicode_data_file, output_file, do_lower_case):

  """
    This generates mappings needed by the GPU for various codepoints.

    For each unicode code-point, This generates a c/cpp header file with a vector where
    v[codepoint] = mapping.

    With one codepoint->mapping pair per line

    The mapping is split as follows:

    bit range -> Name:description

    0 - 20 -> New code point: 
              
               The the code point of the character obtained after lowercasing
               and stripping accents from the original character. 

               In some cases, multiple characters are needed for decomposition
               (hangul characters and some other special characters). For those 
               cases, bit 29 indicates that we need more characters and a second 
               lookup is needed in an auxilary table. 

    21 - 22 -> (# bytes - 1) needed for the new character          
    
    23 -> 1 if decomposition needs more multiple and 0 otherwise. If 1, we need to 
            do a lookup into an auxilary table by code-point to get the remaining 
            characters from the decomposition. From running through the entire range
            of code-points, we decompose to at most 3 characters and since an 8 bytes
            would be needed to store both code-points, the utf-8 encoding of the 
            next characters are stored next to each other in an 8 byte array.
    
    24 - 25 -> Tokenization category: This is a number from 0 to 3 with the following
                                      meaning:

        0 -> add " " left and right of char (Needed for punctuation and chinese)
        1 -> character should be padded after lower casing
        2 -> character should always be replaced
        3 -> character should be removed if lower casing
        4 -> character should always be replaced (Only used for white space characters currently) 
        5 -> unmapped


    For auxilary characters, we need the following table:
    aux[code_point] = new_chars.

    new_chars is broken down as follows:
     0-31 -> utf-8 encoding of the next character in the decomposition. 0 if no character follows
    32-63 -> utf-8 encoding of the last character in the decomposition. 0 if no character follows


    Params
    --------
    unicode_data_file: str
      The UnicodeData.txt file with all the information about unicode

    output_file: str
      The name of the file where the needed mapping will be stored

    do_lower_case: bool
      A boolean indicating whether the transforms should lower case AND strip accents from
      input strings. (This is how BERT defined lower case so I adopt the same convention)

  """

  unique = set()
  cp_to_map = [0] * (MAX_CODE_POINT + 1)
  aux_table = [0] * (MAX_CODE_POINT + 1)

  max_multi_char_cp = 0
  
  with open(output_file, mode='w') as out:

    for original_cp in range(0, 0x110000):
      original_char = chr(original_cp)

      tokenization_cat = get_tokenization_cat(original_char)
      transformed_cps = _transform_cp(original_char, do_lower_case)
      multi_decomposition = 1 if len(transformed_cps) > 1 else 0

      bytes_less_1 = get_bytes_for_codepoint(transformed_cps[0]) - 1

      if unicodedata.category(original_char) in ["Cn", "Co", "Cs"]:
        multi_decomposition = 0
        transformed_cps = [0]

      new_cp = transformed_cps[0]
    
      if len(transformed_cps) == 2:
        # Loop is in order
        max_multi_char_cp = original_cp
        aux_table[original_cp] = transformed_cps[1] << 32
      elif len(transformed_cps) == 3:
        max_multi_char_cp = original_cp
        aux_table[original_cp] = transformed_cps[1] << 32 | transformed_cps[2]
      else:
        assert len(transformed_cps) == 1, "Can decompose to either 1, 2 or 3 characters"

      mapping = pack_data(new_cp, bytes_less_1, multi_decomposition, tokenization_cat)
      unique.add(mapping)
      cp_to_map[original_cp] = mapping

    # print(max_multi_char_cp)  
    write_header(out, cp_to_map, aux_table, max_multi_char_cp)  

  # groups = sum(1 for k, g in groupby(cp_to_map))
  # print("There are {} contiguous blocks with the same value.".format(groups))
  # print("There are", len(unique), "unique values.")

def write_header(file, mapping, aux_table, max_multi_cp):

  file.write("#include <stdint.h>\n")
  file.write("#include <vector>\n\n")

  file.write("#define SHIFT_FOR_NEW_CP {}\n".format(SHIFT_FOR_NEW_CP))
  file.write("#define NEW_CP_MASK {}\n\n".format(hex(NEW_CP_MASK)))
  
  file.write("#define BYTES_LESS_1_SHIFT {}\n".format(SHIFT_FOR_BYTES_LESS_1))
  file.write("#define BYTES_LESS_1_MASK {}\n\n".format(hex(BYTES_LESS_1_MASK)))

  file.write("#define MULTICHAR_SHIFT {}\n".format(SHIFT_FOR_MULTICHAR))
  file.write("#define MULTICHAR_MASK {}\n\n".format(MULTI_CHAR_MASK))
  
  file.write("#define TOKEN_CAT_SHIFT {}\n".format(SHIFT_FOR_TOK_CAT))
  file.write("#define TOKEN_CAT_MASK {}\n".format(TOKEN_CAT_MASK))
  file.write("#define TOKEN_CAT_ADD_SPACE {}\n".format(TOKEN_CAT_ADD_SPACE))
  file.write("#define TOKEN_CAT_ADD_SPACE_IF_LOWER {}\n".format(TOKEN_CAT_ADD_SPACE_IF_LOWER))
  file.write("#define TOKEN_CAT_REMOVE_CHAR {}\n".format(TOKEN_CAT_REMOVE_CHAR))
  file.write("#define TOKEN_CAT_REMOVE_CHAR_IF_LOWER {}\n".format(TOKEN_CAT_REMOVE_CHAR_IF_LOWER))
  file.write("#define TOKEN_CAT_ALWAYS_REPLACE {}\n\n\n".format(TOKEN_CAT_ALWAYS_REPLACE))

  INTS_PER_LINE = 10
  current = 0

  start_info = "std::vector<uint32_t> cp_data({"
  file.write(start_info)

  while current < len(mapping):
    limit = min(current + INTS_PER_LINE, len(mapping) - 1)
    for num in range(current, limit):
      file.write("{:10}, ".format(mapping[num]))
    current += INTS_PER_LINE

    if current < len(mapping) - 1:
      file.write("\n")
      file.write(" "*len(start_info))

  # write last int
  file.write("{:10}".format(mapping[-1]))
  file.write("});\n\n")

  # write auxilary array  
  start_info = "std::vector<uint64_t> aux_data({"
  file.write(start_info)

  current = 0
  while current <= max_multi_cp:
    limit = min(current + INTS_PER_LINE, max_multi_cp)
    for num in range(current, limit):
      file.write("{:20}, ".format(aux_table[num]))
    current += INTS_PER_LINE

    if current < max_multi_cp - 1:
      file.write("\n")
      file.write(" "*len(start_info))

  # write last int
  file.write("{:20}".format(aux_table[max_multi_cp]))
  file.write("});")

def get_tokenization_cat(char):
  
  # padding cases
  if _add_padding(char):
    return 0
  if _pad_if_lowercase(char):
    return 1

  # remove cases
  if _should_remove(char):
    return 2
  if _remove_if_lowercase(char):
    return 3
  
  # always replace
  if _is_whitespace(char):
    return 4

  return 5

def pack_data(new_char, bytes_less_1, multi_decomposition, tokenization_cat):
  
  assert new_char < 2**21, "New new_char not in 21 bits"

  assert bytes_less_1 <= 3, " bytes_less_1 can be at most 3"

  assert multi_decomposition == 0 or multi_decomposition == 1, "Multi decomposition must be 0 or 1"

  assert tokenization_cat <= 5, "The tokenization category can be at most 5"

  packed_data = (new_char  << SHIFT_FOR_NEW_CP| 
                 bytes_less_1 << SHIFT_FOR_BYTES_LESS_1 |
                 multi_decomposition << SHIFT_FOR_MULTICHAR | 
                 tokenization_cat << SHIFT_FOR_TOK_CAT)

  assert 0 <= packed_data < 2**32, "Packed data must fit in 32 bits"

  return packed_data

def get_bytes_for_codepoint(cp):
  """ Computes the bytes required to store this codepoint in utf-8 
      Based on the utf-8 RFC found here https://tools.ietf.org/html/rfc3629 """

  if 0 <= cp <= 0x7F:
    return 1
  
  if 0x80 <= cp <= 0x7FF:
    return 2

  if 0x800 <= cp <= 0xFFFF:
    return 3
  
  if 0x10000 <= cp <= 0x10FFFF:
    return 4
  
  raise(ValueError("Code point not in valid range [0, 1114111]"))

  
def _add_padding(char):
  return _is_chinese_char(char) or _is_punctuation(char)

def _is_whitespace(char):
  """Checks whether `char` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  # Any character that python would split on is also considered a whitespace
  # character

  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return char.isspace()


def _should_remove(char):
  """Checks whether a character should be removed when tokenizing"""

  # These are technically control characters but BERT treats these as
  # whitespaces

  if char == "\t" or char == "\n" or char == "\r":
    return False

  cp = ord(char)
  if cp == 0 or cp == 0xfffd:
    return True

  cat = unicodedata.category(char)
  if cat in ("Cc", "Cf"):
    return True

  return False

def _remove_if_lowercase(char):
  return unicodedata.category(char) == "Mn"

def  _pad_if_lowercase(char):
  l_of_cps = _transform_cp(char, True)
  return len(l_of_cps) == 1 and _is_punctuation(chr(l_of_cps[0]))

def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""

  cp = ord(char)
  # BERT treats all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but BERT treats them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False

def _is_chinese_char(char):

  """Checks whether CP is the codepoint of a CJK character."""
  # This defines a "chinese character" as anything in the CJK Unicode block:
  #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
  #
  # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
  # despite its name. The modern Korean Hangul alphabet is a different block,
  # as is Japanese Hiragana and Katakana. Those alphabets are used to write
  # space-separated words, so they are not treated specially and handled
  # like the all of the other languages.
  cp = ord(char)
  if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
      (cp >= 0x3400 and cp <= 0x4DBF) or  #
      (cp >= 0x20000 and cp <= 0x2A6DF) or  #
      (cp >= 0x2A700 and cp <= 0x2B73F) or  #
      (cp >= 0x2B740 and cp <= 0x2B81F) or  #
      (cp >= 0x2B820 and cp <= 0x2CEAF) or
      (cp >= 0xF900 and cp <= 0xFAFF) or  #
      (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
    return True

  return False

def _transform_cp(char, do_lower_case):
  """Returns the code-point after lowercasing and stripping an accent OR
      The code-point of the empty space if a unicode whitespace character
      
      Returns empty list if the character does not need to be transformed. 
      
      We do this once and off chip for all characters based on the needs of
      a tokenizer. The normalization algorithm requires recursion which would
      not be ideal on the GPU. """

  if _is_whitespace(char):
    return [ord(" ")]

  if do_lower_case:
    lowered_and_decomp = unicodedata.normalize("NFD", char.lower())       

    new_chars = []
    for c in lowered_and_decomp:
      cat = unicodedata.category(c)
      if cat != "Mn":
        new_chars.append(c)
      
    if not new_chars:
      return [ord(" ")]
    
    if [char] == new_chars:
      return [0]

    return [ord(c) for c in new_chars]
  
  return [0]

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Parse unicode data to extract information needed for GPU")
  parser.add_argument("--unicode_data", default='./UnicodeData.txt', dest="unicode_data", type=str)
  parser.add_argument("--out_file", default="./include/cp_data.h", dest="output_file", type=str)
  parser.add_argument("--do_lower_case", "-l", default=True, dest="do_lower_case", type=bool)


  ns = parser.parse_args()
  categoryList = ["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po", 
                  "Lu", "Ll", "Lt", "Lm", "Lo",
                  "Mn", "Mc", "Me",
                  "Nd", "Nl", "No",
                  "Sm", "Sc", "Sk", "So",
                  "Zs", "Zl", "Zp",
                  "Cc", "Cf", "Cs", "Co", "Cn"]

MAX_ALLOWED_CATEGORIES = 32 
assert len(categoryList) < MAX_ALLOWED_CATEGORIES

gen_metadata(ns.unicode_data, ns.output_file, ns.do_lower_case)
