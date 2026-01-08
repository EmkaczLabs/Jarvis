# ruff: noqa: RUF001, RUF002
import re
from typing import ClassVar


class SpokenTextConverter:
    """
    A utility class for converting text containing numbers, dates, times, and currency
    into their spoken-word equivalents. Can you inagine how many edge cases you have to cover?

    This class provides methods to normalize and convert various text elements, such as:
    - Numbers (e.g., "3.14" → "three point one four")
    - Dates (e.g., "1/1/2024" → "one/one/twenty twenty-four")
    - Times (e.g., "3:00pm" → "three o'clock")
    - Currency (e.g., "$50.00" → "fifty dollars")
    - Percentages (e.g., "50%" → "fifty percent")
    - Titles and abbreviations (e.g., "Mr." → "Mister")
    - Years (e.g., "1999" → "nineteen ninety-nine")
    - Large numbers (e.g., "1000000" → "one million")
    - Decimals (e.g., "0.5" → "zero point five")
    - Mixed text (e.g., "The meeting is at 3:00pm on 1/1/2024.")
    - And more...


    Example usage:
        >>> converter = SpokenTextConverter()
        >>> result = converter.convert_to_spoken_text("The meeting is at 3:00pm on 1/1/2024.")
        >>> print(result)
        The meeting is at three o'clock on one/one/twenty twenty-four.
    """

    CONTRACTIONS: ClassVar[dict[str, str]] = {
        "I'm": "I am",
        "I'll": "I will",
        "I've": "I have",
        "I'd": "I would",
        "won't": "will not",
        "can't": "cannot",
        "n't": " not",
        "'ll": " will",
        "'re": " are",
        "'ve": " have",
        "'m": " am",
        "'d": " would",
        "ain't": "is not",
    }

    def __init__(self) -> None:
        # Initialize any necessary state or configurations here, maybe for other languages?

        # Precompile quick check pattern
        # Note: Only check for mathematical operators that aren't commonly used in regular text
        """
        Initialize the SpokenTextConverter with regex patterns for identifying convertible text content.

        This method sets up a compiled regular expression pattern to quickly identify text
        that may require conversion, such as numbers, currency symbols, mathematical operators,
        common abbreviations, and ellipses.

        The pattern checks for:
        - Digits
        - Currency symbols ($ and £)
        - Specific mathematical operators (multiplication, division, exponentiation, roots)
        - Common title abbreviations
        - Ellipses (three or more dots, including spaced versions)

        The regex is compiled with verbose mode (re.VERBOSE) to allow more readable pattern construction.
        """
        self.convertible_pattern = re.compile(
            r"""(?x)
            \d                        # Any digit
            |\$|£                     # Currency symbols
            |[×÷^√∛]                 # Unambiguous mathematical operators (removed hyphen)
            |\b(?:Dr|Mr|Mrs|Ms)\.    # Common abbreviations
            |\.{3,}|\. \. \.         # Triple dots (including spaced version)
            """
        )

        # TODO: Add compiled regex patterns for other conversions

    def _number_to_words(self, num: float | str) -> str:
        """
        Convert a number into its spoken-word equivalent.

        Handles integers, floating-point numbers, and numeric strings, including:
        - Negative numbers
        - Large numbers (e.g., millions, billions)
        - Decimal numbers
        - Zero and whole numbers

        Parameters:
            num (float | str): The number to convert. Can be an integer, float, or numeric string.

        Returns:
            str: The spoken-word representation of the number.

        Raises:
            ValueError: If the input cannot be converted to a valid number.

        Examples:
            >>> converter._number_to_words(42)
            'forty-two'
            >>> converter._number_to_words(-17)
            'negative seventeen'
            >>> converter._number_to_words(1234567)
            'one million two hundred thirty-four thousand five hundred sixty-seven'
            >>> converter._number_to_words(3.14)
            'three point one four'
        """
        try:
            if isinstance(num, str):
                # Check if it's actually an integer in string form
                if "." not in num or num.endswith(".0"):
                    num = int(float(num))
                else:
                    num = float(num)

            # Special handling for integers
            if isinstance(num, int) or (isinstance(num, float) and num.is_integer()):
                num = int(num)  # Convert to int if it's a whole number

            if num == 0:
                return "zero"

            ones = [
                "zero",
                "one",
                "two",
                "three",
                "four",
                "five",
                "six",
                "seven",
                "eight",
                "nine",
                "ten",
                "eleven",
                "twelve",
                "thirteen",
                "fourteen",
                "fifteen",
                "sixteen",
                "seventeen",
                "eighteen",
                "nineteen",
            ]
            tens = [
                "",
                "",
                "twenty",
                "thirty",
                "forty",
                "fifty",
                "sixty",
                "seventy",
                "eighty",
                "ninety",
            ]
            scales = ["", "thousand", "million", "billion"]

            def process_chunk(n: int, scale: int) -> str:
                """
                Convert a chunk of a number into its spoken word representation.

                This method handles converting a three-digit number chunk into words, including handling
                hundreds, tens, and ones places. It supports numbers from 0 to 999 and can append
                scale words (thousand, million, etc.) when appropriate.

                Parameters:
                    n (int): The number chunk to convert (0-999)
                    scale (int): The scale index representing the magnitude (0 for ones, 1 for thousands,
                    2 for millions, etc.)

                Returns:
                    str: The spoken word representation of the number chunk, including optional scale word

                Example:
                    process_chunk(123, 1) returns "one hundred twenty-three thousand"
                    process_chunk(45, 0) returns "forty-five"
                """
                if n == 0:
                    return ""

                hundreds = n // 100
                remainder = n % 100

                words = []

                if hundreds > 0:
                    words.append(f"{ones[hundreds]} hundred")

                if remainder > 0:
                    if remainder < 20:
                        words.append(ones[remainder])
                    else:
                        tens_digit = remainder // 10
                        ones_digit = remainder % 10
                        if ones_digit == 0:
                            words.append(tens[tens_digit])
                        else:
                            words.append(f"{tens[tens_digit]}-{ones[ones_digit]}")

                if scale > 0 and len(words) > 0:
                    words.append(scales[scale])

                return " ".join(words)

            # Handle negative numbers
            if num < 0:
                return "negative " + self._number_to_words(abs(num))

            # Handle whole numbers differently from decimals
            if isinstance(num, int):
                if num == 0:
                    return "zero"

                intermediate_result: list[str] = []
                scale = 0

                while num > 0:
                    chunk = num % 1000
                    if chunk != 0:
                        chunk_words = process_chunk(chunk, scale)
                        intermediate_result.insert(0, chunk_words)
                    num //= 1000
                    scale += 1

                return " ".join(filter(None, intermediate_result))
            else:
                # Handle decimal numbers
                str_num = f"{num:.10f}".rstrip("0")  # Handle floating point precision
                if "." in str_num:
                    int_part, dec_part = str_num.split(".")
                else:
                    int_part, dec_part = str_num, ""

                int_num = int(int_part)

                # Convert integer part
                if int_num == 0:
                    result = "zero"
                else:
                    intermediate_result = []
                    scale = 0
                    while int_num > 0:
                        chunk = int_num % 1000
                        if chunk != 0:
                            chunk_words = process_chunk(chunk, scale)
                            intermediate_result.insert(0, chunk_words)
                        int_num //= 1000
                        scale += 1
                    result = " ".join(filter(None, intermediate_result))

                # Add decimal part if it exists
                if dec_part:
                    result = result + " point " + " ".join(ones[int(digit)] for digit in dec_part)
                return result
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid number format: {num}") from e

    def _split_num(self, num: re.Match) -> str:
        """
        Convert numbers, times, and years into their spoken-word equivalents.

        This method handles complex conversions for:
        - Time formats (12-hour and 24-hour)
            - With or without AM/PM
            - Handles "o'clock" for zero minutes
            - Converts minutes less than 10 with "oh"
        - Year formats
            - Single years (e.g., 1999)
            - Decades (e.g., 1950s)
            - Special handling for 2000 and 2000s
            - Supports plural forms for decades

        Parameters:
            num (re.Match): A regex match object containing a time, year, or number string.

        Returns:
            str: The spoken-word equivalent of the input time, year, or number.

        Raises:
            ValueError: If the input cannot be parsed as a valid time or number.
        """
        try:
            match_str = num.group()
            if ":" in match_str:
                # Split out any AM/PM first
                time_str = match_str.lower()
                am_pm = ""
                if "am" in time_str:
                    am_pm = " a m"
                    time_str = time_str.replace("am", "").strip()
                elif "pm" in time_str:
                    am_pm = " p m"
                    time_str = time_str.replace("pm", "").strip()

                try:
                    h, m = [int(n) for n in time_str.split(":")]
                    if not (0 <= h <= 23 and 0 <= m <= 59):
                        return match_str

                    # Handle minutes based on whether we have AM/PM
                    if m == 0:
                        if am_pm:  # If we have AM/PM, just use the hour
                            time = f"{self._number_to_words(h)}"
                        else:  # No AM/PM, use o'clock
                            time = f"{self._number_to_words(h)} o'clock"
                    elif m < 10:
                        time = f"{self._number_to_words(h)} oh {self._number_to_words(m)}"
                    else:
                        time = f"{self._number_to_words(h)} {self._number_to_words(m)}"

                    return f"{time}{am_pm}"

                except ValueError:
                    return match_str

            # Year handling
            try:
                number = int(match_str.rstrip("s"))  # Remove 's' if present
                if len(match_str) == 4 or (len(match_str) == 5 and match_str.endswith("s")):
                    left, right = divmod(number, 100)
                    s = "s" if match_str.endswith("s") else ""

                    # Special case for 2000 and 2000s
                    if number == 2000:
                        if s:
                            return "twenty hundreds"
                        else:
                            return "two thousand"
                    elif right == 0:
                        return f"{self._number_to_words(left)} hundred{s}"
                    elif right < 10:
                        return f"{self._number_to_words(left)} oh {self._number_to_words(right)}{s}"
                    else:
                        # Handle plural for decades (e.g., 1950s → "nineteen fifties")
                        if s and right >= 10:
                            decade_word = self._number_to_words(right).replace(" ", "-")
                            if decade_word.endswith("y"):
                                decade_word = decade_word[:-1] + "ies"
                            else:
                                decade_word += "s"
                            return f"{self._number_to_words(left)} {decade_word}"
                        return f"{self._number_to_words(left)} {self._number_to_words(right)}{s}"

                return self._number_to_words(number)
            except ValueError:
                return match_str
        except Exception:
            return num.group()

    def _flip_money(self, m: re.Match[str]) -> str:
        """
        Convert currency expressions into their spoken-word equivalents.

        Handles currency conversions for dollars and pounds, including whole numbers and decimal
        amounts. Supports singular and plural forms, and manages edge cases like zero cents/pence.

        Parameters:
            m (re.Match[str]): A regex match object containing a currency expression (e.g., "$50.00")

        Returns:
            str: The spoken-word representation of the currency amount

        Raises:
            ValueError: If the currency format is invalid or cannot be parsed

        Examples:
            "$5.00" → "five dollars"
            "$1.50" → "one dollar and fifty cents"
            "£10.00" → "ten pounds"
            "£1.01" → "one pound and one penny"
        """
        try:
            m = m.group()
            if not m or len(m) < 2:
                raise ValueError("Invalid currency format")

            bill = "dollar" if m[0] == "$" else "pound"
            amount_str = m[1:]

            if amount_str.isalpha():
                return f"{self._number_to_words(int(amount_str))} {bill}s"
            elif "." not in amount_str:
                amount = int(amount_str)
                s = "" if amount == 1 else "s"
                return f"{self._number_to_words(amount)} {bill}{s}"

            try:
                b, c = amount_str.split(".")
                if not b:  # Handle case like "$.50"
                    b = "0"
                s = "" if b == "1" else "s"
                c = int(c.ljust(2, "0"))

                # Don't add cents/pence if it's zero
                if c == 0:
                    return f"{self._number_to_words(int(b))} {bill}{s}"

                coins = f"cent{'' if c == 1 else 's'}" if m[0] == "$" else ("penny" if c == 1 else "pence")
                return f"{self._number_to_words(int(b))} {bill}{s} and {self._number_to_words(c)} {coins}"
            except ValueError as e:
                raise ValueError(f"Invalid currency format: {m}") from e
        except Exception:
            return m  # Return original text if conversion fails

    def _point_num(self, num: re.Match[str]) -> str:
        """
        Convert a decimal number to its spoken-word representation.

        Parameters:
            num (re.Match[str]): A regex match object containing a decimal number.

        Returns:
            str: The spoken-word representation of the decimal number.

        Examples:
            "3.14" → "three point one four"
            "0.99" → "zero point nine nine"
            "-2.5" → "negative two point five"
        """
        try:
            num = num.group()
            if not num:
                return num

            # Split into whole and fractional parts
            if "." in num:
                whole, frac = num.split(".", 1)
            else:
                whole, frac = num, ""

            # Convert whole part
            if whole:
                whole_part = self._number_to_words(float(whole))
            else:
                whole_part = ""

            # Convert fractional part
            if frac:
                # Special handling for single-digit fractions
                if len(frac) == 1:
                    frac_part = self._number_to_words(int(frac))
                else:
                    frac_part = " ".join(self._number_to_words(int(digit)) for digit in frac)
            else:
                frac_part = ""

            # Combine whole and fractional parts
            if whole_part and frac_part:
                return f"{whole_part} point {frac_part}"
            elif whole_part:
                return whole_part
            else:
                return frac_part
        except Exception:
            return num.group()

    def _number_to_ordinal(self, n: int) -> str:
        """Convert an integer to its ordinal spoken-word equivalent (e.g., 1 -> first, 18 -> eighteenth)."""
        ordinals_small = {
            0: "zeroth",
            1: "first",
            2: "second",
            3: "third",
            4: "fourth",
            5: "fifth",
            6: "sixth",
            7: "seventh",
            8: "eighth",
            9: "ninth",
            10: "tenth",
            11: "eleventh",
            12: "twelfth",
            13: "thirteenth",
            14: "fourteenth",
            15: "fifteenth",
            16: "sixteenth",
            17: "seventeenth",
            18: "eighteenth",
            19: "nineteenth",
            20: "twentieth",
        }
        if n in ordinals_small:
            return ordinals_small[n]

        # Handle teen exceptions (11-13) which always use 'th'
        if 10 < (n % 100) < 14:
            return f"{self._number_to_words(n)}th"

        words = self._number_to_words(n)
        # Determine the last word (handles hyphens)
        if "-" in words:
            base, last = words.rsplit("-", 1)
            sep = "-"
        elif " " in words:
            base, last = words.rsplit(" ", 1)
            sep = " "
        else:
            base, last = "", words
            sep = ""

        unit_map = {
            "one": "first",
            "two": "second",
            "three": "third",
            "five": "fifth",
            "eight": "eighth",
            "nine": "ninth",
            "twelve": "twelfth",
        }

        if last in unit_map:
            new_last = unit_map[last]
        else:
            new_last = last + "th"

        if base:
            return f"{base}{sep}{new_last}"
        return new_last

    def convert_to_spoken_text(self, text: str) -> str:
        """
        Convert the given text to its spoken-word equivalent using a regex-based tokenizer
        that preserves text and converts numbers, ordinals, times, currencies and percentages.
        """
        try:
            # Quick check for empty or whitespace-only text
            if not text or text.isspace():
                return ""

            # Normalize newlines and excessive spaces
            text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

            # Regex to find interesting numeric/time/currency tokens
            token_re = re.compile(
                r"(\$\s*\d{1,3}(?:[,\d]*)(?:\.\d+)?|£\s*\d{1,3}(?:[,\d]*)(?:\.\d+)?|\d{1,2}:\d{2}(?:[ap]m)?|\b\d{4}s?\b|\b\d+(?:\.\d+)?%?\b|\b\d+(?:st|nd|rd|th)\b)",
                flags=re.IGNORECASE,
            )

            def _repl(m: re.Match[str]) -> str:
                tok = m.group(0)

                # Currency
                if tok.lstrip().startswith("$") or tok.lstrip().startswith("£"):
                    try:
                        return self._flip_money(re.match(r".*", tok))
                    except Exception:
                        return tok

                # Time
                if ":" in tok:
                    mm = re.match(r"^\d{1,2}:\d{2}(?:[ap]m)?$", tok, flags=re.IGNORECASE)
                    if mm:
                        return self._split_num(mm)
                    return tok

                # Year (e.g., 1999 or 1950s)
                if re.fullmatch(r"\d{4}s?", tok):
                    mm = re.match(r"^\d{4}s?$", tok)
                    return self._split_num(mm)

                # Percentage
                if tok.endswith("%"):
                    num = tok[:-1]
                    try:
                        return f"{self._number_to_words(num)} percent"
                    except Exception:
                        return tok

                # Ordinal like 18th
                m_ord = re.match(r"^(\d+)(st|nd|rd|th)$", tok, flags=re.IGNORECASE)
                if m_ord:
                    n = int(m_ord.group(1))
                    try:
                        return self._number_to_ordinal(n)
                    except Exception:
                        return tok

                # Plain number/decimal
                if re.fullmatch(r"\d+(?:\.\d+)?", tok):
                    try:
                        return self._number_to_words(tok)
                    except Exception:
                        return tok

                return tok

            result = token_re.sub(_repl, text)

            # Post-processing: Handle specific cases like "a m" / "p m" to "am" / "pm"
            result = re.sub(r"\b(a m|p m)\b", lambda x: x.group().replace(" ", ""), result)

            return result
        except Exception:
            return text

    def text_to_spoken(self, text: str) -> str:
        """Backward-compatible alias for convert_to_spoken_text used throughout the codebase and tests."""
        return self.convert_to_spoken_text(text)

    def _convert_percentages(self, text: str) -> str:
        """Convert percentage expressions (e.g., '50%') to spoken form (e.g., 'fifty percent').

        This helper is kept for unit tests and historical callers that expect a dedicated
        percentage conversion method.
        """
        try:
            def repl(m: re.Match[str]) -> str:
                num = m.group(1)
                try:
                    spoken = self._number_to_words(num)
                except Exception:
                    spoken = num
                return f"{spoken} percent"

            return re.sub(r"(\d+(?:\.\d+)?)\s*%", repl, text)
        except Exception:
            return text
