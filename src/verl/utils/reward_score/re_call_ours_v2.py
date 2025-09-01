import re
import string
from typing import Union, List
from collections import Counter

# v2 进行的调整：主要让模型关注在回答形式是否正确；
# format分三点进行打分，每一点0.1分
# 然后再新增f1 score

def remove_micro_tool_calls_and_responses(response: str):
    """
    Extracts the content from <answer> and removes <micro_tool_call> and <micro_response> tags
    to return the final model response.
    
    Args:
        response (str): The full response string containing <answer>, <micro_tool_call>, and <micro_response>.

    Returns:
        str: The model's final response without the micro tool calls and responses.
    """
    # Step 1: Extract content from <answer> tag
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if not answer_match:
        return None  # If no <answer> tag found
    
    answer_content = answer_match.group(1)
    
    # Step 2: Remove <micro_tool_call>...</micro_tool_call> and <micro_response>...</micro_response> content
    answer_content = re.sub(r'<micro_tool_call>.*?</micro_tool_call>', '', answer_content, flags=re.DOTALL)
    answer_content = re.sub(r'<micro_response>.*?</micro_response>', '', answer_content, flags=re.DOTALL)
    answer_content = re.sub(r'\s+', ' ', answer_content).strip()
    # Step 3: Return the cleaned answer content
    return answer_content.strip()



def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if s is None:
        return ""
    else:
        return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])
    
    return final_metric['f1']

def validate_template_format(text: str) -> tuple[float, str]:
    """
    validate the template format
    return: (format_score, error message, last_response)
    """
    # extract all assistant responses
    assistant_responses = []
    current_pos = 0
    response_return = ""
    while True: # 先提取所有模型生成的回复
        start_pos = text.find("<|im_start|>assistant\n", current_pos)
        if start_pos == -1:
            break
        end_pos = text.find("<|im_end|>", start_pos)
        if end_pos == -1:
            break
        response = text[start_pos + len("<|im_start|>assistant\n"):end_pos].strip()
        assistant_responses.append(response)
        current_pos = end_pos + len("<|im_end|>")


    if not assistant_responses:
        return 0, "no assistant response found", response_return


    # 最后根据这三个flag True的个数得到总分
    macro_tool_call_format_flag = True
    micro_tool_call_format_flag = True
    key_info_save_format_flag = True

    error_message = ""

    # 统计一下这三类工具的使用次数，如果为0肯定也返回False
    final_answer_count = 0
    final_macro_tool_call_count = 0
    final_micro_tool_call_count = 0
    final_key_info_save_count = 0

    for response in assistant_responses:
        # 先判断是思考还是回答的response
        think_count = response.count("<think>")

        if think_count != 0:
            think_end_count = response.count("</think>")
            # 1. 判断个数是否一致
            if think_count != think_end_count:
                # 最基础的错误，后面都不用看了
                return 0, f"<think> and </think> are not paired: think={think_count}, think_end={think_end_count}", response_return

            # 然后检查宏观检索部分以及关键信息保存部分
            # check <tool_call> and </tool_call> pair
            macro_tool_call_count = response.count("<macro_tool_call>")
            macro_tool_call_end_count = response.count("</macro_tool_call>")
            if macro_tool_call_count != macro_tool_call_end_count:
                return 0, f"<macro_tool_call> and </macro_tool_call> are not paired: macro_tool_call={macro_tool_call_count}, macro_tool_call_end={macro_tool_call_end_count}", response_return
            final_macro_tool_call_count += macro_tool_call_count

            key_info_save_count = response.count("<key_info_save>")
            key_info_save_end_count = response.count("</key_info_save>")
            if key_info_save_count != key_info_save_end_count:
                return 0, f"<key_info_save> and </key_info_save> are not paired: key_info_save={key_info_save_count}, key_info_save_end={key_info_save_end_count}", response_return
            final_key_info_save_count += key_info_save_count


            # 3. check the content of each tool_call can be parsed as json
            current_pos = 0
            while True and macro_tool_call_format_flag:
                tool_call_start = response.find("<macro_tool_call>", current_pos)
                if tool_call_start == -1:
                    break
                tool_call_end = response.find("</macro_tool_call>", tool_call_start)
                if tool_call_end == -1:
                    break
                
                tool_call_content = response[tool_call_start + len("<macro_tool_call>"):tool_call_end].strip()
                
                # check if it contains name and arguments
                if '"name"' not in tool_call_content or '"arguments"' not in tool_call_content:
                    macro_tool_call_format_flag = False
                    # return False, "tool_call is missing name or arguments field"
                
                try:
                    import json
                    json.loads(tool_call_content)
                except json.JSONDecodeError:
                    macro_tool_call_format_flag = False
                    # return False, f"tool_call is not a valid json: {tool_call_content}"
                
                current_pos = tool_call_end + len("</macro_tool_call>")


            current_pos = 0
            while True and key_info_save_format_flag:
                tool_call_start = response.find("<key_info_save>", current_pos)
                if tool_call_start == -1:
                    break
                tool_call_end = response.find("</key_info_save>", tool_call_start)
                if tool_call_end == -1:
                    break
                
                tool_call_content = response[tool_call_start + len("<key_info_save>"):tool_call_end].strip()

                try:
                    import json
                    json.loads(tool_call_content)
                except json.JSONDecodeError:
                    key_info_save_format_flag = False
                    # return False, f"tool_call is not a valid json: {tool_call_content}"
                
                current_pos = tool_call_end + len("</key_info_save>")


        answer_count = response.count("<answer>")
        if answer_count != 0:
            answer_end_count = response.count("</answer>")
            if answer_count != answer_end_count:
                return 0, f"<answer> and </answer> are not paired: answer={answer_count}, answer_end={answer_end_count}", response_return
            final_answer_count += answer_count
            # 然后检查微观检索部分
            micro_tool_call_count = response.count("<micro_tool_call>")
            micro_tool_call_end_count = response.count("</micro_tool_call>")
            if micro_tool_call_count != micro_tool_call_end_count:
                return 0, f"<micro_tool_call> and </micro_tool_call> are not paired: micro_tool_call={micro_tool_call_count}, micro_tool_call_end={micro_tool_call_end_count}", response_return
            final_micro_tool_call_count += micro_tool_call_count


            current_pos = 0
            while True and micro_tool_call_format_flag:
                tool_call_start = response.find("<micro_tool_call>", current_pos)
                if tool_call_start == -1:
                    break
                tool_call_end = response.find("</micro_tool_call>", tool_call_start)
                if tool_call_end == -1:
                    break
                
                tool_call_content = response[tool_call_start + len("<micro_tool_call>"):tool_call_end].strip()

                try:
                    import json
                    json.loads(tool_call_content)
                except json.JSONDecodeError:
                    micro_tool_call_format_flag = False
                    # return False, f"tool_call is not a valid json: {tool_call_content}"
                
                current_pos = tool_call_end + len("</micro_tool_call>")

        if think_count + answer_count == 0:
            # 如果既没有思考也没有回答，格式错误
            return 0, f"<think> 和 <answer> tag 都没有", response_return



    if final_answer_count == 0: # 从来没有出现过answer tag
        return 0, "missing <answer> tag", response_return

    # 从来没有调用过相关工具
    if final_key_info_save_count == 0:
        key_info_save_format_flag = False
    if final_macro_tool_call_count == 0:
        macro_tool_call_format_flag = False
    if final_micro_tool_call_count == 0:
        micro_tool_call_format_flag = False


    format_score = 0
    reason = ""
    if macro_tool_call_format_flag:
        format_score += 0.1
        reason += "macro_tool_call_format is Ture. "
    else:
        format_score += 0  # This line is optional because it's redundant. `format_score` remains unchanged if `False`.
        reason += "macro_tool_call_format is False. "
    
    if key_info_save_format_flag:
        format_score += 0.2
        reason += "key_info_save_format is Ture. "
    else:
        format_score += 0  # This line is optional because it's redundant. `format_score` remains unchanged if `False`.
        reason += "key_info_save_format is False. "

    if micro_tool_call_format_flag:
        format_score += 0.3
        reason += "micro_tool_call_format is Ture. "
    else:
        format_score += 0  # This line is optional because it's redundant. `format_score` remains unchanged if `False`.
        reason += "micro_tool_call_format is False. "


    # # 4. check if the last response contains \\boxed
    # if "\\box" not in assistant_responses[-1]:
    #     return False, "the last response is missing \\boxed"
    response = assistant_responses[-1]
    return format_score, reason, response


# 从这里开始，solution_str是模型生成的答案，ground_truth是标准答案
def compute_score_with_format(tokenizer, solution_str, ground_truth) -> tuple[float, str]:
    if not solution_str.endswith(tokenizer.eos_token): # 先检查格式是否正确
        print(f'not end with eos token')
        return 0, f'not end with eos token'

    
    # 2. 验证文本格式，例如是否有think、tool_call等标签
    format_score = 0

    # 首先验证宏观检索的相关部分
    valid_template_score, reason, response = validate_template_format(solution_str)
    print(f"format score is {valid_template_score}, reason is: {reason}")

    if valid_template_score == 0:
        print("格式不正确，format reward 为 0！")
        return 0
    else:
        format_score = valid_template_score


    print("格式部分正确，计算答案的F1 score！")


    try:
        answer = remove_micro_tool_calls_and_responses(response)

        # answer = remove_boxed(last_boxed_only_string(response)) # 去除返回答案中的\boxed{}，就是把答案取出来
    except Exception as e:
        return valid_template_score, f'答案提取错误 error: {e}'


    # 3. answer_score
    f1_score = get_f1_score(answer, ground_truth)
    print(f"模型完整的思维链是：{solution_str}")
    print()
    print(f"answer is {answer}, ground_truth is {ground_truth}")
    print(f"f1 score is {f1_score}")
    print()
    # if f1_score > 0:
    #     return f1_score, f'correct answer, get f1 score: {f1_score}'
    # else:
    #     return 0.1, f'wrong answer but good format: {answer}' # 如果 F1 分数为 0，说明文本生成不准确，此时会给一个较低的奖励分数，奖励格式正确（例如 0.1），


    total_score = format_score + f1_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {f1_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")

    return total_score