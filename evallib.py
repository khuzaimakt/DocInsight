from typing import List, Mapping, Tuple

import pandas as pd
from bert_score import score as bert_score
from langchain.prompts.chat import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from torchmetrics.text.rouge import ROUGEScore

from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import (
    ContextRelevancyEvaluator,
)

from datasets import Dataset 
from ragas.metrics import context_precision,context_recall,context_relevancy,context_entity_recall
from ragas import evaluate

import nest_asyncio
from tqdm.asyncio import tqdm_asyncio

nest_asyncio.apply()

import pandas as pd


def evaluate_responses_with_overlap(data: pd.DataFrame,
                                    response_col: str,
                                    reference_response_col: str)->pd.DataFrame:
    rouge_scorer = ROUGEScore(rouge_keys=("rouge1", "rouge2", "rougeL"))
    rouge_scores = [
        rouge_scorer(preds=response, target=reference)
        for (response, reference) in
        zip(data[response_col].values, data[reference_response_col].values)
    ]

    rouge_scores_df = pd.DataFrame(rouge_scores)
    for col in rouge_scores_df.columns:
        rouge_scores_df[col] = rouge_scores_df[col].apply(lambda x: x.item())

    rouge_scores_df.set_index(data.index, inplace=True)
    rouge_scores_df["response_length"] = data[response_col].str.len()
    rouge_scores_df["length_ratio"] = (
        rouge_scores_df["response_length"] /
        data[reference_response_col].str.len())

    return rouge_scores_df


def evaluate_responses_with_overlap_bert(data: pd.DataFrame,
                                    response_col: str,
                                    reference_response_col: str)->pd.DataFrame:
    """
    Evaluates text responses using ROUGE and BERTScore metrics.

    Parameters:
    - data (pd.DataFrame): DataFrame containing text responses and references.
    - response_col (str): Column name for the responses.
    - reference_response_col (str): Column name for the reference responses.

    Returns:
    - pd.DataFrame: DataFrame containing calculated ROUGE and BERTScore metrics
    """

    if response_col not in \
    data.columns or \
    reference_response_col not in \
    data.columns:
        raise ValueError("Specified columns are not in the DataFrame")

    responses = data[response_col].tolist()
    references = data[reference_response_col].tolist()

    rouge_scorer = ROUGEScore(rouge_keys=("rouge1", "rouge2", "rougeL"))

    rouge_scores = [
        rouge_scorer(preds=response, target=reference)
        for response, reference in zip(data[response_col].values,
                                       data[reference_response_col].values)
    ]
    rouge_scores_df = pd.DataFrame(rouge_scores)
    rouge_scores_df = rouge_scores_df.applymap(lambda x: x.item())

    rouge_scores_df.set_index(data.index, inplace=True)
    rouge_scores_df["response_length"] = data[response_col].str.len()
    rouge_scores_df["length_ratio"] = (
        rouge_scores_df["response_length"] /
        data[reference_response_col].str.len())

    # Calculate BERT scores
    precision, recall, f1_score = bert_score(responses, references,
                                             lang="en", verbose=True)
    bert_scores_df = pd.DataFrame({
        'bertscore_precision': precision.numpy(),
        'bertscore_recall': recall.numpy(),
        'bertscore_f1': f1_score.numpy()
    })
    bert_scores_df.set_index(data.index, inplace=True)

    # Combine and return scores
    combined_scores_df = pd.concat([rouge_scores_df, bert_scores_df], axis=1)

    return combined_scores_df

# def run_default_llm_based_evaluation(data: pd.DataFrame,
#                            eval_llm: ChatOpenAI | ChatAnthropic,
#                            query_col: str,
#                            response_col: str,
#                            reference_response_col: str | None = None,
#                            context_col: str | None = None) -> pd.DataFrame:
#     evaluations = evaluate_responses_with_llm_multi(
#         data, eval_llm,
#         [eval_prompts.ACCURACY_EVALUATION_PROMPT,
#          eval_prompts.APTNESS_EVALUATION_PROMPT,
#          eval_prompts.TONE_EVALUATION_PROMPT],
#          ["accuracy", "aptness", "tone"],
#          query_col, response_col, reference_response_col, context_col)
#     response_lengths = data[response_col].apply(len)
#     evaluations["response_length"] = response_lengths
#     return evaluations


def evaluate_responses_with_llm_multi(data: pd.DataFrame,
                             eval_llm: ChatOpenAI | ChatAnthropic,
                             eval_prompt_templates: List[ChatPromptTemplate],
                             eval_names: List[str],
                             query_col: str,
                             response_col:str,
                             reference_response_col: str | None = None,
                             context_col: str | None = None,) -> pd.DataFrame:
    templates_and_names = zip(eval_prompt_templates, eval_names)
    eval_results = [
        evaluate_responses_with_llm(data, eval_llm,
                                    eval_prompt_template,
                                    query_col, response_col,
                                    reference_response_col,
                                    context_col, eval_name=eval_name)
        for eval_prompt_template, eval_name in templates_and_names]
    eval_results = pd.concat(eval_results, axis=1)
    return eval_results

def evaluate_responses_with_llm(data: pd.DataFrame,
                       eval_llm: ChatOpenAI | ChatAnthropic,
                       eval_prompt_template: ChatPromptTemplate,
                       query_col: str,
                       response_col:str,
                       reference_response_col: str | None = None,
                       context_col: str | None = None,
                       eval_name: str | None = None) -> pd.DataFrame:
    use_reference_response = (
        "reference_response"
        in eval_prompt_template.input_variables)
    use_context = "context" in eval_prompt_template.input_variables

    result = data.apply(
        lambda row: evaluate_one_response_with_llm(
            eval_llm, eval_prompt_template,
            row[query_col], row[response_col],
            row[reference_response_col] if use_reference_response else None,
            row[context_col] if use_context else None), axis=1)
    if eval_name is not None:
        result.columns = [f"{col}_{eval_name}" for col in result.columns]

    return result

def evaluate_one_response_with_llm(eval_llm: ChatOpenAI | ChatAnthropic,
                          eval_prompt_template: ChatPromptTemplate,
                          query: str,
                          response: str,
                          reference_response: str | None = None,
                          context: str | None = None) -> pd.Series:
    prompt_dict = {
        "query": query,
        "response": response
    }

    if reference_response is not None:
        prompt_dict["reference_response"] = reference_response
    if context is not None:
        prompt_dict["context"] = context

    complete_prompt = eval_prompt_template.format_messages(**prompt_dict)
    eval_result = eval_llm.invoke(complete_prompt)

    feedback, score = [
        item.strip()
        for item in eval_result.content.split("[RESULT]")
    ]

    return pd.Series({"feedback": feedback, "score": score})

def calculate_answered_status(responses: pd.Series) -> pd.Series:
    return responses.apply(lambda response: not is_decline_to_answer(response))

_DECLINE_TO_ANSWER_STRING = (
    "I could not find the answer to your question in the provided document.")
def is_decline_to_answer(response: str) -> bool:
    return response.strip().count(_DECLINE_TO_ANSWER_STRING) > 0

def add_column_with_new_responses(
        data: pd.DataFrame,
        query_col: str,
        source_doc_col: str,
        new_col_name: str,
        query_response_mapping: Mapping[Tuple[str, str], str]):
    """Utility method to add a new column of responses to an existing
    data frame.

    Args:
        - data: the data frame to add the new column to.
        - query_col: the name of the column in the data frame that
          contains queries
        - source_doc_col: the name of the column in the data frame that
          contains the source document
        - new_col_name: the name of the new column to add to the data frame
        - query_response_mapping: a dictionary mapping (source_doc, query)
          pairs to responses
    """
    data[new_col_name] = data.apply(
        lambda row:
        query_response_mapping[
            (row[source_doc_col], row[query_col])], axis=1)
    return


def evaluate_chunking_ragas(data: pd.DataFrame,
                                    response_col: str,
                                    query_col: str,
                                    contexts_col,
                                    ground_truth_col:str)->pd.DataFrame:
    
    dataframe_list = []

    for index,row in data.iterrows():
        
        data_samples = {
        'question': [f"{row[query_col]}"],
        'answer': [f"{row[response_col]}"],
        'contexts' : [[f"{row[contexts_col]}"]],
        'ground_truth': [f"{row[ground_truth_col]}"]
        }
        dataset = Dataset.from_dict(data_samples)
        score = evaluate(dataset,metrics=[context_precision,context_recall,context_relevancy,context_entity_recall])
        

        dataframe_list.append(score.to_pandas())

    final_dataframe = pd.concat(dataframe_list, ignore_index=True)

    return final_dataframe


async def evaluate_chunking_llama_index(data: pd.DataFrame,
                                    response_col: str,
                                    query_col: str,
                                    contexts_col,
                                    ground_truth_col:str)->pd.DataFrame:
    
    context_relevancy_evaluator = ContextRelevancyEvaluator(
    llm=OpenAI(temperature=0, model="gpt-4"),
    )

    eval_tasks = []

    for index,row in data.iterrows():
        
        if isinstance(f"{row[contexts_col]}", list):
            context_list = [str(c) for c in f"{row[contexts_col]}"]
        else:
            context_list = [str(f"{row[contexts_col]}")]

        data_samples = {
        'question': f"{row[query_col]}",
        'answer': f"{row[response_col]}",
        'contexts' : context_list,
        'ground_truth': f"{row[ground_truth_col]}"
        }


        eval_tasks.append(
            context_relevancy_evaluator.aevaluate(
                query=data_samples['question'],
                contexts=data_samples['contexts'],
            )
        )

        eval_results = await tqdm_asyncio.gather(*eval_tasks)

        data = [
        {
            'query': result.query,
            'response': result.response,
            'feedback': result.feedback,
            'score': result.score
        }
        for result in eval_results
        ]

        df_eval = pd.DataFrame(data)

        return df_eval