from openai import *
import jieba
from rank_bm25 import BM25Okapi
import ast
import copy

class Sync_LLM_Model:
    def __init__(self, base_url="Your URL", model="Your Selected Model"):
        self.query_client = OpenAI(
            api_key="Your API Key",
            base_url=base_url,
            max_retries=3,
            timeout=60.0
        )
        self.model = model

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        messages = [{'role': 'system', 'content': system_prompt}]
        messages.append({'role': 'user', 'content': user_prompt})
        try:
            completion = self.query_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                timeout=300
            )
            return completion.choices[0].message.content
        except RateLimitError:
            return "The API call frequency has exceeded the limit. Please try again later."
        except APITimeoutError:
            return "The request timed out, please try again."
        except APIError as e:
            print(f"API Error occurred: {str(e)}")
            return "API call exception! Please re-enter."
        except Exception as e:
            print(f"Unexpected error occurred: {str(e)}")
            return "The question is abnormal! Please re-enter."

def data_load_after(file_path):
    i = 1
    data_dic = {"Privacy":[],"NoPrivacy":[]}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            templeline = ast.literal_eval(line.strip())
            i = i + 1
            data_dic["Privacy"] = data_dic["Privacy"] + templeline["Privacy"]
            data_dic["NoPrivacy"] = data_dic["NoPrivacy"] + templeline["NoPrivacy"]
    return data_dic

def rag_data_load(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for templeline in file.readlines():
            templeline = templeline.strip()
            data_list.append(templeline)
    return data_list

def update_list(i, j, A, B, k=0):

    if (i + 1) % 20 == 1:
        A.clear()
        A.extend(B[:j])
        return A
    else:
        for idx in range(j):
            if idx < len(A):
                A[idx] += B[idx]

        if (i + 1) % 20 == 0 and k == 1:
            return A.index(max(A))

    return A

def main():
    datapre = data_load_after("Dataset/Language.jsonl")
    data = datapre
    Yeslist = []
    Nolist = []
    Indexyes = 0
    Indexno = 0
    Correctnumber = 0
    Yescorrect = 0
    Nocorrect = 0
    Qianwen = Sync_LLM_Model()

    for i in range(0,2000):

        dataprivacy = data["Privacy"][i]
        datanoprivacy = data["NoPrivacy"][i]
        system_prompt = "You are an expert in named entity recognition and you are very sensitive to entity information, such as a person's name, address, IP address, data, time or similar data. Your principle is that You need to identify these entity information in a text as much as possible, and achieve a relatively good recognition accuracy while identifying the entity information as much as possible!!!"
        user_prompt = f"""You are an expert in named entity recognition and you are very sensitive to entity information, such as a person's name, address, IP address, data, time or similar data. Your principle is that You need to identify these entity information in a text as much as possible, and achieve a relatively good recognition accuracy while identifying the entity information as much as possible!!!
        Here are some examples of Identifications you can follow to learn how to identity the named entity information and improve the accuracy of your Identification:\n
        [Text for Identification]
        Regarding the morning schedule, Bowen has been assigned the first slot at 10:20am. The candidate should prepare all necessary documentation and arrive at the main conference room for the comprehensive evaluation process.\n
        [Process of Identification]
        From the name "Bowen" and the keywords: "morning schedule", "first slot" and time "10:20am".
        We can know that this is a text for the Bowen meeting arrangement. So the named entities include the person name "Bowen", the time is "10:20am", and the scenario is "Meeting Arrangement".\n
        [Output]
        ["Name":" Bowen", "Time":" 10:20am", "Location": "Main conference room", "Scenario": "Meeting Arrangement"]\n
        [Text for Identification]
        Our office is located at 11 Mirage Street, for inquiries please call the 142-167-982 phone number.\n
        [Process of Identification]
        From the keywords "Our office", "inquiries", and the number "142-167-982", we can see that this text is a notification to the public. So the named entities include the department name "office", the address is "11 Mirage Street", the phone number is "142-167-982" and the scenario is "public notification".\n
        [Output]
        ["Department": "office", "Location":" 11 Mirage Street", "Phone Number": "142-167-982", "Scenario": "Office public notification"]\n
        Tell me the named entities in the text in the form of a List, It contains the category and specific information of the named entity, Just like the output in the example, Please output directly in the form of a list without output identifiers and other line breaks. Again: Your principle is that You need to identify these entity information in a text as much as possible, and achieve a relatively good recognition accuracy while identifying the entity information as much as possible. Please adhere strictly to this principle!!! Below is the Text for Identification.\n
        [Text for Identification]\n
        {dataprivacy}"""

        response = Qianwen.generate_response(system_prompt, user_prompt)
        response = response.replace(".", "")
        privacyres = response

        system_prompt = "You are an expert in named entity recognition and you are very sensitive to entity information, such as a person's name, address, IP address, data, time or similar data. Your principle is that You need to identify these entity information in a text as much as possible, and achieve a relatively good recognition accuracy while identifying the entity information as much as possible!!!"
        user_prompt = f"""You are an expert in named entity recognition and you are very sensitive to entity information, such as a person's name, address, IP address, data, time or similar data. Your principle is that You need to identify these entity information in a text as much as possible, and achieve a relatively good recognition accuracy while identifying the entity information as much as possible!!!
        Here are some examples of Identifications you can follow to learn how to identity the named entity information and improve the accuracy of your Identification:\n
        [Text for Identification]
        Regarding the morning schedule, Bowen has been assigned the first slot at 10:20am. The candidate should prepare all necessary documentation and arrive at the main conference room for the comprehensive evaluation process.\n
        [Process of Identification]
        From the name "Bowen" and the keywords: "morning schedule", "first slot" and time "10:20am".
        We can know that this is a text for the Bowen meeting arrangement. So the named entities include the person name "Bowen", the time is "10:20am", and the scenario is "Meeting Arrangement".\n
        [Output]
        ["Name":" Bowen", "Time":" 10:20am", "Location": "Main conference room", "Scenario": "Meeting Arrangement"]\n
        [Text for Identification]
        Our office is located at 11 Mirage Street, for inquiries please call the 142-167-982 phone number.\n
        [Process of Identification]
        From the keywords "Our office", "inquiries", and the number "142-167-982", we can see that this text is a notification to the public. So the named entities include the department name "office", the address is "11 Mirage Street", the phone number is "142-167-982" and the scenario is "public notification".\n
        [Output]
        ["Department": "office", "Location":" 11 Mirage Street", "Phone Number": "142-167-982", "Scenario": "Office public notification"]\n
        Tell me the named entities in the text in the form of a List, It contains the category and specific information of the named entity, Just like the output in the example, Please output directly in the form of a list without output identifiers and other line breaks. Again: Your principle is that You need to identify these entity information in a text as much as possible, and achieve a relatively good recognition accuracy while identifying the entity information as much as possible. Please adhere strictly to this principle!!! Below is the Text for Identification.\n
        [Text for Identification]\n
        {datanoprivacy}"""

        response = Qianwen.generate_response(system_prompt, user_prompt)
        response = response.replace(".", "")
        noprivacyres = response

        corpus = rag_data_load("Knowledge Base/JudgeText.txt")
        corpus_yes = rag_data_load("Knowledge Base/JudgeText-privacy.txt")
        corpus_no = rag_data_load("Knowledge Base/JudgeText-noprivacy.txt")
        Ragprocess = rag_data_load("Knowledge Base/JudgeProcess.txt")
        Ragoutput = rag_data_load("Knowledge Base/JudgeResult.txt")
        Ragentity = rag_data_load("Knowledge Base/NER.txt")
        Ragprocess_yes = rag_data_load("Knowledge Base/JudgeProcess-privacy.txt")
        Ragentity_yes = rag_data_load("Knowledge Base/NER-privcy.txt")
        Ragprocess_no = rag_data_load("Knowledge Base/JudgeProcess-noprivacy.txt")
        Ragentity_no = rag_data_load("Knowledge Base/NEU-noprivacy.txt")

        tokenized_corpus_yes_update = [list(jieba.cut(doc)) for doc in corpus_yes]
        bm25_yes = BM25Okapi(tokenized_corpus_yes_update)
        queryyes = dataprivacy
        queryno = datanoprivacy
        tokenized_query_privacy = list(jieba.cut(queryyes))
        tokenized_query_noprivacy = list(jieba.cut(queryno))
        doc_scores_privacy_yes = list(bm25_yes.get_scores(tokenized_query_privacy))
        doc_scores_noprivacy_yes = list(bm25_yes.get_scores(tokenized_query_noprivacy))

        tokenized_corpus_no_update = [list(jieba.cut(doc)) for doc in corpus_no]
        bm25_no = BM25Okapi(tokenized_corpus_no_update)
        doc_scores_privacy_no = list(bm25_no.get_scores(tokenized_query_privacy))
        doc_scores_noprivacy_no = list(bm25_no.get_scores(tokenized_query_noprivacy))
        if (i+1) % 20 == 1:
            Yeslist.clear()
            Nolist.clear()
            Yeslength = len(corpus_yes)
            Nolength = len(corpus_no)
            Yeslist = update_list(i,Yeslength,Yeslist,doc_scores_privacy_yes)
            Yeslist = update_list(i, Yeslength, Yeslist, doc_scores_noprivacy_yes)
            Nolist = update_list(i, Nolength, Nolist, doc_scores_privacy_no)
            Nolist = update_list(i, Nolength, Nolist, doc_scores_noprivacy_no)
        elif (i+1) % 20 == 0:
            Yeslist = update_list(i, Yeslength, Yeslist, doc_scores_privacy_yes)
            Indexyes = update_list(i, Yeslength, Yeslist, doc_scores_noprivacy_yes,1)
            Nolist = update_list(i, Nolength, Nolist, doc_scores_privacy_no)
            Indexno = update_list(i, Nolength, Nolist, doc_scores_noprivacy_no,1)
        else:
            Yeslist = update_list(i,Yeslength,Yeslist,doc_scores_privacy_yes)
            Yeslist = update_list(i, Yeslength, Yeslist, doc_scores_noprivacy_yes)
            Nolist = update_list(i, Nolength, Nolist, doc_scores_privacy_no)
            Nolist = update_list(i, Nolength, Nolist, doc_scores_noprivacy_no)

        tokenized_corpus = [list(jieba.cut(doc)) for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        doc_scores_privacy = list(bm25.get_scores(tokenized_query_privacy))

        tokenized_corpus_yes_judge = [list(jieba.cut(corpus_yes[Indexyes]))]
        bm25_yesjudge = BM25Okapi(tokenized_corpus_yes_judge)
        doc_scores_privacy_yesjudge = bm25_yesjudge.get_scores(tokenized_query_privacy)
        YesJudgePrivacy = doc_scores_privacy_yesjudge[0]

        tokenized_corpus_no_judge = [list(jieba.cut(corpus_no[Indexno]))]
        bm25_nojudge = BM25Okapi(tokenized_corpus_no_judge)
        doc_scores_privacy_nojudge = bm25_nojudge.get_scores(tokenized_query_privacy)
        NoJudgePrivacy = doc_scores_privacy_nojudge[0]
        doc_scores_privacy_temple = copy.deepcopy(doc_scores_privacy)
        filter_scores_privacy = [value for value in doc_scores_privacy if value != YesJudgePrivacy and value != NoJudgePrivacy]
        first_max_privacy = max(filter_scores_privacy)
        filter_scores_privacy.remove(first_max_privacy)
        second_max_privacy = max(filter_scores_privacy)
        w = doc_scores_privacy.index(first_max_privacy)
        w1 = doc_scores_privacy.index(second_max_privacy)

        doc_scores_noprivacy = list(bm25.get_scores(tokenized_query_noprivacy))
        doc_scores_privacy_yesjudge = bm25_yesjudge.get_scores(tokenized_query_noprivacy)
        YesJudgenoPrivacy = doc_scores_privacy_yesjudge[0]
        doc_scores_privacy_nojudge = bm25_nojudge.get_scores(tokenized_query_noprivacy)
        NoJudgenoPrivacy = doc_scores_privacy_nojudge[0]
        doc_scores_noprivacy_temple = copy.deepcopy(doc_scores_noprivacy)
        filter_scores_noprivacy = [value for value in doc_scores_noprivacy if value != YesJudgenoPrivacy and value != NoJudgenoPrivacy]
        first_max_noprivacy = max(filter_scores_noprivacy)
        filter_scores_noprivacy.remove(first_max_noprivacy)
        second_max_noprivacy = max(filter_scores_noprivacy)
        z = doc_scores_noprivacy.index(first_max_noprivacy)
        z1 = doc_scores_noprivacy.index(second_max_noprivacy)

        for k in range(0,2):
            YesCountnumber = 0
            NoCountnumber = 0
            Answerlist = [" "," "," "," "]
            Confidencylist = [0,0,0,0]
            if k == 0:
                IndexRag = w
                IndexRag1 = w1
                Judgedata = dataprivacy
                Judgeentity = privacyres
                doc_scores_temple = doc_scores_privacy_temple
                Score_temple1 = YesJudgePrivacy
                Score_temple2 = NoJudgePrivacy
            else:
                IndexRag = z
                IndexRag1 = z1
                Judgedata = datanoprivacy
                Judgeentity = noprivacyres
                doc_scores_temple = doc_scores_noprivacy_temple
                Score_temple1 = YesJudgenoPrivacy
                Score_temple2 = NoJudgenoPrivacy

            system_prompt = " You are an expert in text field or scene analysis. You can analyze the field or scene to which the text belongs based on the keywords in the text. Your job is to determine whether the given texts belong to the same field or scene. "
            user_prompt = f""" You are an expert in text field or scene analysis. You can analyze the field or scene to which the text belongs based on the keywords in the text. Your job is to determine whether the given texts belong to the same field or scene.
            [Text 1]
            {corpus_yes[Indexyes]}\n
            [Text 2]
            {corpus_no[Indexno]}\n
            Please tell me directly whether the Text for judgement belongs to the same field or scenario as Text 1 or Text 2. There are only two options, Yes and No. Please directly and only output Yes or No.
            [Text for judgement]
            {Judgedata}"""
            response = Qianwen.generate_response(system_prompt, user_prompt)
            response = response.replace(".", "")

            if (doc_scores_temple[IndexRag] + doc_scores_temple[IndexRag1]) < (Score_temple1 + Score_temple2) and response == "Yes":
                system_prompt = "You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. Your principle is that you can incorrectly classify non-private data as private and never classify private data as non-private, ensure a high percentage of correctness on this basis. Please adhere strictly to this principle!!!"
                user_prompt = f"""You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private.
                Here are some examples of judgements you can follow to learn how to judge private text and improve the accuracy of your judgements:\n
                {corpus_yes[Indexyes]}
                [Keywords]
                {Ragentity_yes[Indexyes]}
                [Process of judgement]
                {Ragprocess_yes[Indexyes]}
                [Output]
                Yes\n
                [Text for judgement]
                {corpus_no[Indexno]}
                [Keywords]
                {Ragentity_no[Indexno]}
                [Process of judgement]
                {Ragprocess_no[Indexno]}
                [Output]
                No\n
                Tell me directly whether it is a privacy text and output the answer, Just like the output in the example. There are only two options, Yes and No. Please directly and only output Yes or No.
                [Text for judgement]
                {Judgedata}
                [Keywords]
                {Judgeentity}"""
                response = Qianwen.generate_response(system_prompt, user_prompt)
                response = response.replace(".", "")
                if response == "Yes" and k == 0:
                    Correctnumber = Correctnumber + 1
                    Yescorrect = Yescorrect + 1
                elif response == "No" and k == 1:
                    Correctnumber = Correctnumber + 1
                    Nocorrect = Nocorrect + 1

            else:
                system_prompt = "You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. Your principle is that you can incorrectly classify non-private data as private and never classify private data as non-private, ensure a high percentage of correctness on this basis. Please adhere strictly to this principle!!!"
                user_prompt = f"""You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. 
                {corpus_yes[Indexyes]}
                [Keywords]
                {Ragentity_yes[Indexyes]}
                [Process of judgement]
                {Ragprocess_yes[Indexyes]}
                [Output]
                Yes\n
                [Text for judgement]
                {corpus_no[Indexno]}
                [Keywords]
                {Ragentity_no[Indexno]}
                [Process of judgement]
                {Ragprocess_no[Indexno]}
                [Output]
                No\n
                Tell me directly whether it is a privacy text and output the answer, Just like the output in the example. There are only two options, Yes and No. Please directly and only output Yes or No.
                [Text for judgement]
                {Judgedata}
                [Keywords]
                {Judgeentity}"""

                response = Qianwen.generate_response(system_prompt,user_prompt)
                response = response.replace(".","")
                Answerlist[0] = response
                if response == "Yes":
                    YesCountnumber = YesCountnumber + 1
                elif response == "No":
                    NoCountnumber = NoCountnumber + 1

                system_prompt = "You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. Your principle is that you can incorrectly classify non-private data as private and never classify private data as non-private, ensure a high percentage of correctness on this basis. Please adhere strictly to this principle!!!"
                user_prompt = f"""You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. 
                Here are some examples of judgements you can follow to learn how to judge private text and improve the accuracy of your judgements:\n
                {corpus_yes[Indexyes]}
                [Keywords]
                {Ragentity_yes[Indexyes]}
                [Process of judgement]
                {Ragprocess_yes[Indexyes]}
                [Output]
                Yes\n
                [Text for judgement]
                {corpus_no[Indexno]}
                [Keywords]
                {Ragentity_no[Indexno]}
                [Process of judgement]
                {Ragprocess_no[Indexno]}
                [Output]
                No\n
                [Text for judgement]\n{corpus[IndexRag]}\n
                [Keywords]\n{Ragentity[IndexRag]}\n
                [Process of judgement]\n{Ragprocess[IndexRag]}\n
                [Output]\n{Ragoutput[IndexRag]}\n
                Tell me directly whether it is a privacy text and output the answer, Just like the output in the example. There are only two options, Yes and No. Please directly and only output Yes or No.
                [Text for judgement]
                {Judgedata}
                [Keywords]
                {Judgeentity}"""

                response = Qianwen.generate_response(system_prompt,user_prompt)
                response = response.replace(".","")
                Answerlist[1] = response
                if response == "Yes":
                    YesCountnumber = YesCountnumber + 1
                elif response == "No":
                    NoCountnumber = NoCountnumber + 1

                system_prompt = "You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. Your principle is that you can incorrectly classify non-private data as private and never classify private data as non-private, ensure a high percentage of correctness on this basis. Please adhere strictly to this principle!!!"
                user_prompt = f"""You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private.
                Here are some examples of judgements you can follow to learn how to judge private text and improve the accuracy of your judgements:\n
                {corpus_yes[Indexyes]}
                [Keywords]
                {Ragentity_yes[Indexyes]}
                [Process of judgement]
                {Ragprocess_yes[Indexyes]}
                [Output]
                Yes\n
                [Text for judgement]
                {corpus_no[Indexno]}
                [Keywords]
                {Ragentity_no[Indexno]}
                [Process of judgement]
                {Ragprocess_no[Indexno]}
                [Output]
                No\n
                [Text for judgement]\n{corpus[IndexRag1]}\n
                [Keywords]\n{Ragentity[IndexRag1]}\n
                [Process of judgement]\n{Ragprocess[IndexRag1]}\n
                [Output]\n{Ragoutput[IndexRag1]}\n
                Tell me directly whether it is a privacy text and output the answer, Just like the output in the example. There are only two options, Yes and No. Please directly and only output Yes or No.
                [Text for judgement]
                {Judgedata}
                [Keywords]
                {Judgeentity}"""

                response = Qianwen.generate_response(system_prompt,user_prompt)
                response = response.replace(".","")
                Answerlist[2] = response
                if response == "Yes":
                    YesCountnumber = YesCountnumber + 1
                elif response == "No":
                    NoCountnumber = NoCountnumber + 1

                system_prompt = "You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. Your principle is that you can incorrectly classify non-private data as private and never classify private data as non-private, ensure a high percentage of correctness on this basis. Please adhere strictly to this principle!!!"
                user_prompt = f"""You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private.
                Here are some examples of judgements you can follow to learn how to judge private text and improve the accuracy of your judgements:\n
                {corpus_yes[Indexyes]}
                [Keywords]
                {Ragentity_yes[Indexyes]}
                [Process of judgement]
                {Ragprocess_yes[Indexyes]}
                [Output]
                Yes\n
                [Text for judgement]
                {corpus_no[Indexno]}
                [Keywords]
                {Ragentity_no[Indexno]}
                [Process of judgement]
                {Ragprocess_no[Indexno]}
                [Output]
                No\n
                [Text for judgement]\n{corpus[IndexRag]}\n
                [Keywords]\n{Ragentity[IndexRag]}\n
                [Process of judgement]\n{Ragprocess[IndexRag]}\n
                [Output]\n{Ragoutput[IndexRag]}\n
                [Text for judgement]\n{corpus[IndexRag1]}\n
                [Keywords]\n{Ragentity[IndexRag1]}\n
                [Process of judgement]\n{Ragprocess[IndexRag1]}\n
                [Output]\n{Ragoutput[IndexRag1]}\n
                Tell me directly whether it is a privacy text and output the answer, Just like the output in the example. There are only two options, Yes and No. Please directly and only output Yes or No.
                [Text for judgement]
                {Judgedata}
                [Keywords]
                {Judgeentity}"""

                response = Qianwen.generate_response(system_prompt,user_prompt)
                response = response.replace(".","")
                Answerlist[3] = response
                if response == "Yes":
                    YesCountnumber = YesCountnumber + 1
                elif response == "No":
                    NoCountnumber = NoCountnumber + 1

                if YesCountnumber > NoCountnumber:
                    if k == 0:
                        Yescorrect = Yescorrect + 1
                        Correctnumber = Correctnumber + 1
                elif NoCountnumber > YesCountnumber:
                    if k == 1:
                        Nocorrect = Nocorrect + 1
                        Correctnumber = Correctnumber + 1
                elif YesCountnumber == NoCountnumber:
                    system_prompt = "You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. Your principle is that you can incorrectly classify non-private data as private and never classify private data as non-private, ensure a high percentage of correctness on this basis. Please adhere strictly to this principle!!!"
                    user_prompt = f"""You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. 
                    Here are some examples of judgements you can follow to learn how to judge private text and improve the accuracy of your judgements:\n
                    {corpus_yes[Indexyes]}
                    [Keywords]
                    {Ragentity_yes[Indexyes]}
                    [Process of judgement]
                    {Ragprocess_yes[Indexyes]}
                    [Output]
                    Yes\n
                    [Text for judgement]
                    {corpus_no[Indexno]}
                    [Keywords]
                    {Ragentity_no[Indexno]}
                    [Process of judgement]
                    {Ragprocess_no[Indexno]}
                    [Output]
                    No\n
                    Evaluate your confidence in the answer by giving an integer from (1-100), please give this integer directly without outputting any of its contents.
                    [Text for judgement]
                    {Judgedata}
                    [Keywords]
                    {Judgeentity}"""
                    response = Qianwen.generate_response(system_prompt, user_prompt)
                    response = response.replace(".", "")
                    try:
                        Confidencylist[0] = int(response)
                    except Exception as e:
                        Confidencylist[0] = 0
                        print("A data type conversion error occurred")

                    system_prompt = "You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. Your principle is that you can incorrectly classify non-private data as private and never classify private data as non-private, ensure a high percentage of correctness on this basis. Please adhere strictly to this principle!!!"
                    user_prompt = f"""You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. 
                    Here are some examples of judgements you can follow to learn how to judge private text and improve the accuracy of your judgements:\n
                    {corpus_yes[Indexyes]}
                    [Keywords]
                    {Ragentity_yes[Indexyes]}
                    [Process of judgement]
                    {Ragprocess_yes[Indexyes]}
                    [Output]
                    Yes\n
                    [Text for judgement]
                    {corpus_no[Indexno]}
                    [Keywords]
                    {Ragentity_no[Indexno]}
                    [Process of judgement]
                    {Ragprocess_no[Indexno]}
                    [Output]
                    No\n
                    [Text for judgement]\n{corpus[IndexRag]}\n
                    [Keywords]\n{Ragentity[IndexRag]}\n
                    [Process of judgement]\n{Ragprocess[IndexRag]}\n
                    [Output]\n{Ragoutput[IndexRag]}\n
                    Evaluate your confidence in the answer by giving an integer from (1-100), please give this integer directly without outputting any of its contents.
                    [Text for judgement]
                    {Judgedata}
                    [Keywords]
                    {Judgeentity}"""
                    response = Qianwen.generate_response(system_prompt, user_prompt)
                    response = response.replace(".", "")
                    try:
                        Confidencylist[1] = int(response)
                    except Exception as e:
                        Confidencylist[1] = 0
                        print("A data type conversion error occurred")

                    system_prompt = "You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. Your principle is that you can incorrectly classify non-private data as private and never classify private data as non-private, ensure a high percentage of correctness on this basis. Please adhere strictly to this principle!!!"
                    user_prompt = f"""You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. 
                    Here are some examples of judgements you can follow to learn how to judge private text and improve the accuracy of your judgements:\n
                    {corpus_yes[Indexyes]}
                    [Keywords]
                    {Ragentity_yes[Indexyes]}
                    [Process of judgement]
                    {Ragprocess_yes[Indexyes]}
                    [Output]
                    Yes\n
                    [Text for judgement]
                    {corpus_no[Indexno]}
                    [Keywords]
                    {Ragentity_no[Indexno]}
                    [Process of judgement]
                    {Ragprocess_no[Indexno]}
                    [Output]
                    No\n
                    [Text for judgement]\n{corpus[IndexRag1]}\n
                    [Keywords]\n{Ragentity[IndexRag1]}\n
                    [Process of judgement]\n{Ragprocess[IndexRag1]}\n
                    [Output]\n{Ragoutput[IndexRag1]}\n
                    Evaluate your confidence in the answer by giving an integer from (1-100), please give this integer directly without outputting any of its contents.
                    [Text for judgement]
                    {Judgedata}
                    [Keywords]
                    {Judgeentity}"""
                    response = Qianwen.generate_response(system_prompt, user_prompt)
                    response = response.replace(".", "")
                    try:
                        Confidencylist[2] = int(response)
                    except Exception as e:
                        Confidencylist[2] = 0
                        print("A data type conversion error occurred")

                    system_prompt = "You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. Your principle is that you can incorrectly classify non-private data as private and never classify private data as non-private, ensure a high percentage of correctness on this basis. Please adhere strictly to this principle!!!"
                    user_prompt = f"""You are a private text inspector. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to classify texts as private and non-private. 
                    Here are some examples of judgements you can follow to learn how to judge private text and improve the accuracy of your judgements:\n
                    {corpus_yes[Indexyes]}
                    [Keywords]
                    {Ragentity_yes[Indexyes]}
                    [Process of judgement]
                    {Ragprocess_yes[Indexyes]}
                    [Output]
                    Yes\n
                    [Text for judgement]
                    {corpus_no[Indexno]}
                    [Keywords]
                    {Ragentity_no[Indexno]}
                    [Process of judgement]
                    {Ragprocess_no[Indexno]}
                    [Output]
                    No\n
                    [Text for judgement]\n{corpus[IndexRag]}\n
                    [Keywords]\n{Ragentity[IndexRag]}\n
                    [Process of judgement]\n{Ragprocess[IndexRag]}\n
                    [Output]\n{Ragoutput[IndexRag]}\n
                    [Text for judgement]\n{corpus[IndexRag1]}\n
                    [Keywords]\n{Ragentity[IndexRag1]}\n
                    [Process of judgement]\n{Ragprocess[IndexRag1]}\n
                    [Output]\n{Ragoutput[IndexRag1]}\n
                    Evaluate your confidence in the answer by giving an integer from (1-100), please give this integer directly without outputting any of its contents.
                    [Text for judgement]
                    {Judgedata}
                    [Keywords]
                    {Judgeentity}"""
                    response = Qianwen.generate_response(system_prompt, user_prompt)
                    response = response.replace(".", "")
                    try:
                        Confidencylist[3] = int(response)
                    except Exception as e:
                        Confidencylist[3] = 0
                        print("A data type conversion error occurred")

                    min_value = min(Confidencylist)
                    min_index = Confidencylist.index(min_value)
                    if Answerlist[min_index] == "No":
                        FinalAnswer = "Yes"
                        if k == 0:
                            Yescorrect = Yescorrect + 1
                            Correctnumber = Correctnumber + 1
                        with open("Knowledge Base/JudgeText.txt", 'a', encoding='utf-8') as f:
                            f.write(Judgedata + "\n")
                            f.close()
                        with open("Knowledge Base/JudgeText-privacy.txt", 'a', encoding='utf-8') as f:
                            f.write(Judgedata + "\n")
                            f.close()
                        if Judgeentity == "":
                            with open("Knowledge Base/NER.txt", 'a', encoding='utf-8') as f:
                                f.write("[]" + "\n")
                                f.close()
                            with open("Knowledge Base/NER-privcy.txt", 'a', encoding='utf-8') as f:
                                f.write("[]" + "\n")
                                f.close()
                        else:
                            with open("Knowledge Base/NER.txt", 'a', encoding='utf-8') as f:
                                f.write(Judgeentity + "\n")
                                f.close()
                            with open("Knowledge Base/NER-privcy.txt", 'a', encoding='utf-8') as f:
                                f.write(Judgeentity + "\n")
                                f.close()
                        with open("Knowledge Base/JudgeResult.txt", 'a', encoding='utf-8') as f:
                            f.write("Yes" + "\n")
                            f.close()

                    if Answerlist[min_index] == "Yes":
                        FinalAnswer = "No"
                        if k == 1:
                            Nocorrect = Nocorrect + 1
                            Correctnumber = Correctnumber + 1
                        with open("Knowledge Base/JudgeText.txt", 'a', encoding='utf-8') as f:
                            f.write(Judgedata + "\n")
                            f.close()
                        with open("Knowledge Base/JudgeText-noprivacy.txt", 'a', encoding='utf-8') as f:
                            f.write(Judgedata + "\n")
                            f.close()
                        if Judgeentity == "":
                            with open("Knowledge Base/NER.txt", 'a', encoding='utf-8') as f:
                                f.write("[]" + "\n")
                                f.close()
                            with open("Knowledge Base/NEU-noprivacy.txt", 'a', encoding='utf-8') as f:
                                f.write("[]" + "\n")
                                f.close()
                        else:
                            with open("Knowledge Base/NER.txt", 'a', encoding='utf-8') as f:
                                f.write(Judgeentity + "\n")
                                f.close()
                            with open("Knowledge Base/NEU-noprivacy.txt", 'a', encoding='utf-8') as f:
                                f.write(Judgeentity + "\n")
                                f.close()
                        with open("Knowledge Base/JudgeResult.txt", 'a', encoding='utf-8') as f:
                            f.write("No" + "\n")
                            f.close()

                    system_prompt = " You are an expert in privacy text analysis and reasoning. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to come up with a detailed analysis and judgment process based on whether the privacy text and known text are privacy text content. "
                    user_prompt = f""" You are an expert in privacy text analysis and reasoning. A text is considered to be private if it contains a person's name, address, IP address, data, time or similar data. Your job is to come up with a detailed analysis and judgment process based on whether the privacy text and known text are privacy text content.
                    Here are some examples of analysis and reasoning you can follow to learn how to generate a specific process for analyzing and determining whether a text is private text:\n
                    [Whether it is a private text]
                    Yes\n
                    [Text for judgement]
                    Regarding the morning schedule, Bowen has been assigned the first slot at 10:20am. The candidate should prepare all necessary documentation and arrive at the main conference room for the comprehensive evaluation process.\n
                    [Keywords]
                    ["Name":" Bowen", "Time":" 10:20am", "Location": "Main conference room", "Scenario": "Meeting Arrangement"]\n
                    [Output]
                    From the name ‘Bowen’ and the keywords: ‘morning schedule’, ‘first slot’ and time ‘10:20am’, we can judge that this text contains private data about personal schedule, so it is a private text.\n
                    [Whether it is a private text]
                    No\n
                    [Text for judgement]
                    Our office is located at 11 Mirage Street, for inquiries please call the 142-167-982 phone number.\n
                    [Keywords]
                    ["Department": "office", "Location":" 11 Mirage Street", "Phone Number": "142-167-982", "Scenario": "Office public notification"]\n
                    [Output]
                    There are no names in the keywords. From the keywords "Our office", "inquiries", and the number "142-167-982", we can see that this text is a notification to the public and does not contain personal privacy. Therefore, this text is non-private text.\n
                    Please generate the analysis and reasoning process in the corresponding language according to the language of the Text for judgement, Just like the Output part in the example. Please output the analysis process text directly without adding line breaks.
                    [Whether it is a private text]
                    {FinalAnswer}
                    [Text for judgement]
                    {Judgedata}
                    [Keywords]
                    {Judgeentity}"""
                    response = Qianwen.generate_response(system_prompt, user_prompt)
                    with open("Knowledge Base/JudgeProcess.txt", 'a', encoding='utf-8') as f:
                        if response == "":
                            f.write("  "+ "\n")
                        else:
                            f.write(response + "\n")
                            f.close()
                    if FinalAnswer == "Yes":
                        with open("Knowledge Base/JudgeProcess-privacy.txt", 'a', encoding='utf-8') as f:
                            if response == "":
                                f.write("  " + "\n")
                            else:
                                f.write(response + "\n")
                                f.close()
                    elif FinalAnswer == "No":
                        with open("Knowledge Base/JudgeProcess-noprivacy.txt", 'a', encoding='utf-8') as f:
                            if response == "":
                                f.write("  " + "\n")
                            else:
                                f.write(response + "\n")
                                f.close()
            if k == 0:
                print(f"The {i + 1}th privacy result has been saved")
            elif k == 1:
                print(f"The {i + 1}th no-privacy result has been saved")
            print(f"Correctnumber:{Correctnumber},Yescorrect{Yescorrect},Nocorrect:{Nocorrect}")
    with open("Result.txt", 'a', encoding='utf-8') as f:
        f.write(f"Correctnumber:{Correctnumber},Yescorrect{Yescorrect},Nocorrect:{Nocorrect}" + "\n")


if __name__ == "__main__":
    main()