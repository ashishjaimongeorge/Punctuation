import pandas as pd
import numpy as np
import difflib as dl
from tqdm import tqdm
import re
import torch
import os
import psutil
import concurrent.futures
import time
import datetime
import multiprocessing
import threading

model, example_texts, languages, punct, apply_te = torch.hub.load(
    repo_or_dir='snakers4/silero-models', model='silero_te'
)

torch._C._jit_set_profiling_mode(False)

start_time = datetime.datetime.now()

def add_punctuation(review_file_path, output_file_path):
    data = pd.read_csv(review_file_path)
    original_data = data.copy()

    data = data[~data['Review'].isnull()].reset_index(drop=True)
    data = data[['Review']].drop_duplicates().reset_index(drop=True)

    bullet_list = ['-', '+', '#', '>', '•']
    splchar_list = ['.', '>', '-', ')', ']', ',', ':', ';']
    spl_numbull_list = ['.', '>']
    bracket_list = [')', '}', ']']

    def bullet_removal(txt):
        cleaned_review = ""
        review = txt
        skip = 0

        if isinstance(review, str):
            for i in range(len(review)):
                if skip > 0:
                    skip -= 1
                elif review[i] == '\n':
                    next_char = i + 1
                    while next_char < len(review) and not review[next_char].isalnum():
                        next_char += 1
                        skip += 1
                    if next_char < (len(review) - 2) and (review[next_char + 1] in splchar_list):
                        if review[next_char + 1] in spl_numbull_list:
                            if review[next_char + 2].isalnum():
                                pass
                            else:
                                skip += 2
                                cleaned_review += ". "
                        else:
                            skip += 2
                            cleaned_review += ". "
                    else:
                        cleaned_review += review[i]
                else:
                    cleaned_review += review[i]

            cleaned_review = re.sub(r'\bPROS?\b\s*\.*', 'Pros ', cleaned_review, flags=re.IGNORECASE)
            cleaned_review = re.sub(r'\bCONS?\b\s*\.*', 'Cons ', cleaned_review, flags=re.IGNORECASE)

        return cleaned_review

    for i in tqdm(range(len(data))):
        data.loc[i, "bullet_point_fix"] = bullet_removal(data.loc[i, "Review"])

    def add_space_after_delimeter(datas: list) -> list:
        new_datas = []
        idx = 0
        for data in datas:
            idx += 1
            data = str(data)
            if len(data) == 0:
                new_datas.append(data)
                continue
            for deli in '.,:;?!%)':
                data_list = data.split(deli)

                data = ""
                count = 0

                for sent in data_list:
                    count += 1
                    if count == 1:
                        data += sent.strip()
                        continue
                    elif (deli == '.' or deli == ',') and len(data) >= 1 and len(sent) >= 1 and data[-1] >= '0' and data[-1] <= '9' and sent[0] >= '0' and sent[0] <= '9':
                        data += deli + sent.strip()
                    elif len(sent) >= 1 and sent[0] == ' ':
                        data += deli + sent.rstrip()
                    elif sent:
                        data += deli + " " + sent.strip()
                    else:
                        data += deli

                data = re.sub(r'(\S)\(', r'\1 (', data)
                data = re.sub(r"(\S)&(\S)", r"\1 & \2", data)

            new_datas.append(data)
        return new_datas

    review_ls = data["bullet_point_fix"]
    data['whitespace sent'] = add_space_after_delimeter(review_ls)

    def apply_te_with_timeout(sentence, lan, timeout=60):
        result = None
        exception = None

        def target():
            nonlocal result, exception
            try:
                result = apply_te(sentence, lan)
            except Exception as e:
                exception = e

        t = threading.Thread(target=target)
        t.start()
        t.join(timeout)

        if t.is_alive():
            raise concurrent.futures.TimeoutError("apply_te function took too long")
        if exception:
            raise exception

        return result

    def process_sentence(sentence):
        try:
            return apply_te_with_timeout(sentence, lan='en', timeout=120)
        except Exception as e:
            return sentence

    def process_data_parallel(data):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_sentence, data.loc[i, "whitespace sent"]): i for i in range(len(data))}

            with open("error_log.txt", "w") as error_log:
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(data)):
                    i = futures[future]
                    try:
                        data.loc[i, "punct_Review"] = future.result()
                    except concurrent.futures.TimeoutError:
                        error_msg = f"TimeoutError at index {i}: {data.loc[i, 'whitespace sent']}\n"
                        error_log.write(error_msg)
                        data.loc[i, "punct_Review"] = data.loc[i, "whitespace sent"]
                    except Exception as e:
                        error_msg = f"Error at index {i}: {e} - {data.loc[i, 'whitespace sent']}\n"
                        error_log.write(error_msg)
                        data.loc[i, "punct_Review"] = data.loc[i, "whitespace sent"]

    process_data_parallel(data)

    def alphanumeric_stretches_and_spans(string):
        pattern = r"[A-Za-z0-9\[\]]+"
        matches = re.finditer(pattern, string)
        tuples_list = []
        for match in matches:
            matched_string = match.group().lower()
            start, end = match.span()
            tuples_list.append((matched_string, (start, end)))
        return tuples_list

    def get_span_replaced_punct_review(punct_rev, orig_rev, pattern):
        span_list = []

        nk_sub_punct_rev = punct_rev.replace(r"##NK]", r"[UNK]")
        nk_sub_punct_rev = re.sub(r"([^U])NK\]", r"\1 [UNK]", nk_sub_punct_rev)
        nk_sub_punct_rev = re.sub(r"\[UNK\](?:\W+\[UNK\])+", r"[UNK]", nk_sub_punct_rev)

        corrected_review = nk_sub_punct_rev

        pr_alphanum_stretches = alphanumeric_stretches_and_spans(nk_sub_punct_rev.lower())
        or_alphanum_stretches = alphanumeric_stretches_and_spans(orig_rev.lower())

        or_word_counter = 0

        multi_unk_for_remove_idx = []
        for i, stretch in enumerate(pr_alphanum_stretches):
            if i == 0:
                continue
            if stretch[0] == '[unk]' and pr_alphanum_stretches[i - 1][0] == '[unk]':
                multi_unk_for_remove_idx.append(i)

        for rem_idx in sorted(multi_unk_for_remove_idx, reverse=True):
            del pr_alphanum_stretches[rem_idx]

        for i in range(len(pr_alphanum_stretches)):
            pr_token_to_check = pr_alphanum_stretches[i][0]
            or_token_to_check = or_alphanum_stretches[or_word_counter][0]
            if pr_token_to_check == or_token_to_check:
                or_word_counter += 1

            if pr_token_to_check == pattern:
                if i < len(pr_alphanum_stretches) - 1:
                    pr_next_word = pr_alphanum_stretches[i + 1][0]
                    if pr_next_word == or_alphanum_stretches[or_word_counter][0]:
                        if or_word_counter > 0:
                            or_replacement_span_begin_idx = or_alphanum_stretches[or_word_counter - 1][1][1]
                            or_replacement_span_end_idx = or_alphanum_stretches[or_word_counter][1][0] - 1
                            replacement_span = orig_rev[or_replacement_span_begin_idx:or_replacement_span_end_idx]
                            span_list.append(replacement_span)
                            continue

                    or_replacement_span_begin_idx = or_alphanum_stretches[or_word_counter][1][0]
                    or_replacement_span_end_idx = or_alphanum_stretches[or_word_counter][1][1]
                    while or_token_to_check != pr_next_word and or_word_counter < len(or_alphanum_stretches) - 1:
                        or_word_counter += 1
                        or_replacement_span_end_idx = or_alphanum_stretches[or_word_counter][1][0] - 1
                        or_token_to_check = or_alphanum_stretches[or_word_counter][0]
                    replacement_span = orig_rev[or_replacement_span_begin_idx:or_replacement_span_end_idx]
                    span_list.append(replacement_span)
                else:
                    or_replacement_span_begin_idx = or_alphanum_stretches[or_word_counter][1][0]
                    or_replacement_span_end_idx = len(orig_rev)
                    replacement_span = orig_rev[or_replacement_span_begin_idx:or_replacement_span_end_idx]
                    span_list.append(replacement_span)

        for rep_span in span_list:
            corrected_review = re.sub(r"\[UNK\]", rep_span, corrected_review, count=1)

        return corrected_review

    def get_non_ascii_list_from_str(ip_str):
        non_ascii_list = []
        pattern = r"[^\x00-\x7F]|&"
        matches = re.finditer(pattern, ip_str)
        for match in matches:
            non_ascii_list.append((match.group(), match.span()[0]))
        return non_ascii_list

    def replace_non_ascii_from_orig(punct_rev, orig_rev):
        orig_rev_spl_chars = get_non_ascii_list_from_str(orig_rev)
        punct_rev_spl_chars = get_non_ascii_list_from_str(punct_rev)
        corrected_review = punct_rev

        if len(orig_rev_spl_chars) != len(punct_rev_spl_chars):
            return orig_rev
        else:
            for spl_char_idx, spl_char in enumerate(orig_rev_spl_chars):
                if punct_rev_spl_chars[spl_char_idx][0] == spl_char[0]:
                    continue
                punct_replace_loc = punct_rev_spl_chars[spl_char_idx][1]
                corrected_review = corrected_review[:punct_replace_loc] + spl_char[0] + corrected_review[punct_replace_loc + 1:]

        return corrected_review

    def fixing_errors(data):
        error_list = ["NK]"]
        both_cnt = 0
        unk_cnt = 0
        ampr_cnt = 0

        for i in tqdm(range(len(data))):
            orig_rev = data.loc[i, "whitespace sent"]
            punct_rev = data.loc[i, "punct_Review"]
            diff = str(orig_rev).count('&') - str(punct_rev).count('&')

            matching_words = [word for word in str(punct_rev).split() if any(item in word for item in error_list)]
            if len(matching_words) > 0 and diff < 0:
                try:
                    unk_fix_review = get_span_replaced_punct_review(punct_rev, orig_rev, "[unk]")
                    data.loc[i, "error_fix_temp"] = replace_non_ascii_from_orig(unk_fix_review, orig_rev)
                except:
                    data.loc[i, "error_fix_temp"] = orig_rev

                data.loc[i, "error_type_b4_ufix"] = "unk and &"
                both_cnt += 1
            elif matching_words:
                try:
                    data.loc[i, "error_fix_temp"] = get_span_replaced_punct_review(punct_rev, orig_rev, "[unk]")
                except:
                    data.loc[i, "error_fix_temp"] = orig_rev

                data.loc[i, "error_type_b4_ufix"] = "unk"
                unk_cnt += 1
            elif diff < 0:
                try:
                    data.loc[i, "error_fix_temp"] = replace_non_ascii_from_orig(punct_rev, orig_rev)
                except:
                    data.loc[i, "error_fix_temp"] = orig_rev

                data.loc[i, "error_type_b4_ufix"] = "&"
                ampr_cnt += 1
            else:
                data.loc[i, "error_fix_temp"] = punct_rev
                data.loc[i, "error_type_b4_ufix"] = "none"

            after_fix_rev = data.loc[i, "error_fix_temp"]
            matching_words_recheck = [word_r for word_r in str(after_fix_rev).split() if any(item_r in word_r for item_r in error_list)]
            diff_recheck = str(orig_rev).count('&') - str(after_fix_rev).count('&')
            if len(matching_words_recheck) > 0 or diff_recheck < 0:
                data.loc[i, "error_fix_temp"] = orig_rev

        print("unk error : ", unk_cnt + both_cnt)
        print("& error : ", both_cnt + ampr_cnt)
        print("amper+unk : ", both_cnt)
        return data

    data = fixing_errors(data)

    def fixing_len_and_string_match(unk_error_df):
        def clean_phrase_(phrase):
            final_phrase = re.sub(r'[^a-zA-Z0-9]', '', phrase)
            return final_phrase

        unk_error_df['Review_AN_only'] = unk_error_df['bullet_point_fix'].apply(lambda x: clean_phrase_(x))
        unk_error_df['error_fix_AN_only'] = unk_error_df['error_fix_temp'].apply(lambda x: clean_phrase_(x))

        unk_error_df.loc[unk_error_df['Review_AN_only'].str.len() == unk_error_df['error_fix_AN_only'].str.len(), 'length_match'] = 'TRUE'
        unk_error_df.loc[unk_error_df['Review_AN_only'].str.len() != unk_error_df['error_fix_AN_only'].str.len(), 'length_match'] = 'False'

        unk_error_df.loc[(unk_error_df['length_match'] == "False"), 'error_fix'] = unk_error_df['whitespace sent']
        unk_error_df.loc[(unk_error_df['length_match'] == "TRUE"), 'error_fix'] = unk_error_df['error_fix_temp']
        return unk_error_df

    data = fixing_len_and_string_match(data)

    def new_line_removal(txt):
        cleaned_review = ""
        review = txt
        skip = 0
        for i in range(len(review)):
            if skip > 0:
                skip -= 1
            elif review[i] == '\n':
                pre_char = i
                while review[pre_char - 1] == " ":
                    pre_char = pre_char - 1

                if review[pre_char - 1].isalnum():
                    cleaned_review += review[i].replace('\n', '. ')
                elif review[pre_char - 1] in bracket_list:
                    cleaned_review += review[i].replace('\n', '. ')
                else:
                    cleaned_review += review[i].replace('\n', ' ')
            else:
                cleaned_review += review[i]

        return cleaned_review

    for i in tqdm(range(len(data))):
        data.loc[i, "new_line_fix"] = new_line_removal(data.loc[i, "error_fix"])

    columns_to_remove = [
        'bullet_point_fix',
        'whitespace sent',
        'punct_Review',
        'error_fix_temp',
        'error_type_b4_ufix',
        'Review_AN_only',
        'error_fix_AN_only',
        'length_match',
        'error_fix'
    ]

    df = data.drop(columns_to_remove, axis=1)

    df = df.rename(columns={'new_line_fix': 'Punctuated_Review'})
    df = df.rename(columns={'Review': 'Original_Review', 'Punctuated_Review': 'Punc_Review'})

    final_file = pd.merge(original_data, df, left_on='Review', right_on='Original_Review', how='left')

    del final_file['Original_Review']

    final_file = final_file.rename(columns={'Review': 'Original_Review', 'Punc_Review': 'Review'})

    final_file.to_csv(output_file_path, index=False)

    end_time = datetime.datetime.now()
    total_time = end_time - start_time

    print("Total time taken: ", end="")
    if total_time.days > 0:
        print(f"{total_time.days} day{'s' if total_time.days > 1 else ''} ", end="")
    if total_time.seconds // 3600 > 0:
        print(f"{total_time.seconds // 3600} hour{'s' if total_time.seconds // 3600 > 1 else ''} ", end="")
    if (total_time.seconds // 60) % 60 > 0:
        print(f"{(total_time.seconds // 60) % 60} minute{'s' if (total_time.seconds // 60) % 60 > 1 else ''} ", end="")
    if total_time.seconds % 60 > 0:
        print(f"{total_time.seconds % 60} second{'s' if total_time.seconds % 60 > 1 else ''} ", end="")
    if total_time.microseconds > 0:
        print(f"{total_time.microseconds} microsecond{'s' if total_time.microseconds > 1 else ''}")

if __name__ == "__main__":
    review_file_path = 'your_input_file.csv'
    output_file_path = 'your_output_file.csv'
    add_punctuation(review_file_path, output_file_path)
