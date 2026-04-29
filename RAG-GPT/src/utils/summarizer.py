from langchain_community.document_loaders import PyPDFLoader
from utils.utilities import count_num_tokens
from utils.load_config import LoadConfig

APPCFG = LoadConfig()


class Summarizer:
    @staticmethod
    def summarize_the_pdf(
        file_dir: str,
        max_final_token: int,
        token_threshold: int,
        gpt_model: str,
        temperature: float,
        summarizer_llm_system_role: str,
        final_summarizer_llm_system_role: str,
        character_overlap: int,
    ) -> str:
        docs = []
        docs.extend(PyPDFLoader(file_dir).load())
        print(f"Document length: {len(docs)}")
        max_summarizer_output_token = int(max_final_token / len(docs)) - token_threshold
        full_summary = ""
        counter = 1
        print("Generating the summary..")

        if len(docs) > 1:
            for i in range(len(docs)):
                if i == 0:
                    prompt = docs[i].page_content + docs[i + 1].page_content[:character_overlap]
                elif i < len(docs) - 1:
                    prompt = (
                        docs[i - 1].page_content[-character_overlap:]
                        + docs[i].page_content
                        + docs[i + 1].page_content[:character_overlap]
                    )
                else:
                    prompt = docs[i - 1].page_content[-character_overlap:] + docs[i].page_content

                role = summarizer_llm_system_role.format(max_summarizer_output_token)
                full_summary += Summarizer.get_llm_response(gpt_model, temperature, role, prompt=prompt)
                print(f"Page {counter} was summarized. ", end="")
                counter += 1
        else:
            full_summary = docs[0].page_content

        print("\nFull summary token length:", count_num_tokens(full_summary, model=gpt_model))
        return Summarizer.get_llm_response(
            gpt_model, temperature, final_summarizer_llm_system_role, prompt=full_summary
        )

    @staticmethod
    def get_llm_response(
        gpt_model: str,
        temperature: float,
        llm_system_role: str,
        prompt: str,
    ) -> str:
        response = APPCFG.openai_client.chat.completions.create(
            model=gpt_model,
            messages=[
                {"role": "system", "content": llm_system_role},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
