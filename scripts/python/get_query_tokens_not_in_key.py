from collections import Counter
from prenlp.tokenizer import Mecab

def get_query_tokens_not_in_key(tokens_q, tokens_k, only_nouns=True, noun_checker=Mecab().tokenizer.nouns):
    """tokens_q 토큰 중 tokens_k 에 존재하지 않는 토큰 리스트 반환 
        :tokens_q: query token list
        :tokens_k: key token list
        :only_nouns: nouns에 대해서만 확인
    """
    counter_q, counter_k = Counter(tokens_q), Counter(tokens_k)
    counter_diff = counter_q - counter_k
    diff_tokens = counter_diff.keys()
    
    if only_nouns:
        diff_nouns = []        
        for token in diff_tokens:
            diff_nouns += noun_checker(token)
        return diff_nouns
    
    else:
        return diff_tokens

if __name__ == '__main__':
    q = ['저', '는', '학교', '에', '가', '고', '있', '어요']
    k = ['저', '는', '수업', '에', '가', '고', '있', '어요']

    diff_tokens = get_query_tokens_not_in_key(q, k)
    print(diff_tokens, len(diff_tokens))
