import re
import unicodedata
from unidecode import unidecode
import emoji
import hgtk

def word_to_jamo(token):
    def to_special_token(jamo):
        if not jamo:
            return '-'
        else:
            return jamo

    decomposed_token = ''
    for char in token:
        if not hgtk.checker.is_hangul(char):
            decomposed_token += char
            continue

        try:
            cho, jung, jong = hgtk.letter.decompose(char)

            # 자모가 빈 문자일 경우 특수문자 -로 대체
            cho = to_special_token(cho)
            jung = to_special_token(jung)
            jong = to_special_token(jong)
            decomposed_token = decomposed_token + cho + jung + jong

        except Exception as exception:
            if type(exception).__name__ == 'NotHangulException':
                decomposed_token += char

    return decomposed_token

def is_hangul(ch):
    codepoint = ord(ch)
    if 0x1100 <= codepoint <= 0x11FF: # Hangul Jamo
        return True
    if 0x3130 <= codepoint <= 0x318F: # Hangul Compatibility Jamo
        return True
    if 0xAC00 <= codepoint <= 0xD7A3: # Hangul Syllables
        return True
    return False

def replace_garbage(line):
    replace_empty = [
        "방금 수신한 문자메시지는 해외에서 발송되었습니다.",
        "(광고)",
        "ifg@",
        "[Web발신]",
        "[국제발신]",
        "[국외발신]"
    ]

    replace_astrik = [
        "권세인",
        "권*인",
        "고문관",
        "고*관",
        "숭실대",
        "숭실",
        "대창고등학교",
        "대창고",
        "1040032145",
        "2145",
        "광고",
        "*"
    ]

    for replace in replace_empty:
        line = line.replace(replace, '')
    for replace in replace_astrik:
        line = line.replace(replace, ' ')
    return line

def skip_message(message):
    skip_words = [
        "한국투자",
        "[한투]",
        "매너콜"
    ]
    for skip_word in skip_words:
        if skip_word in message:
            return True
    return False

def clean_message(item, apply_jamo):
    # 한글 조합형을 완성형으로 합성
    item = unicodedata.normalize('NFC', item)

    c = []
    for ch in item:
        if emoji.is_emoji(ch): # 이모지 제거
            continue
        elif is_hangul(ch): # 한글은 그대로 넣기
            c.append(ch)
        elif unicodedata.category(ch) in ['Cc', 'Cf', 'Cs', 'Co', 'Cn']: # 표시할 수 없는 컨트롤 문자 제거
            c.append(' ')
        else: # 유니코드를 아스키 코드 범위 내로 표현 가능하게 최대한 변환
            decoded = unidecode(ch)
            c.append(decoded)

    cleaned = ''.join(c)
    cleaned = replace_garbage(cleaned) # 개인정보, 메타데이터 치환
    cleaned = re.sub(r' +', ' ', cleaned) # 연속되는 공백 단일공백으로 치환
    cleaned = re.sub(r'\*+', '*', cleaned) # 연속되는 마스킹 문자 단일 마스킹으로 치환
    if apply_jamo:
        cleaned = word_to_jamo(cleaned) # 자모단위 분해
    cleaned = cleaned.lower()
    cleaned = cleaned.strip()

    return cleaned