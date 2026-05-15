"""SEA-LION translation generation."""

from __future__ import annotations

from ..clients.sea_lion_client import SeaLionClient
from ..utils.text import clean_model_output


INDONESIAN_TRANSLATION_GUIDANCE = (
    "English-to-Indonesian educational translation requirements:\n"
    "1. Accuracy: preserve the full source meaning; do not omit, summarize, add examples, or change claims.\n"
    "2. Acceptability: use standard Indonesian that follows EYD Edisi V and KBBI-compatible usage, including "
    "spelling, capitalization, punctuation, absorption words, and di-/ke- affixes versus di/ke prepositions.\n"
    "3. Readability: write clear, natural Indonesian that an educated reader can follow without having to "
    "reconstruct the English source.\n"
    "4. Lexical equivalence: use precise Indonesian terms for science, social science, and education; retain "
    "international terms, formulas, units, and named entities only when that is the standard usage.\n"
    "5. Grammatical equivalence: use natural Indonesian syntax and morphology, including appropriate affixes "
    "such as meN-, di-, ber-, ter-, -kan, and -i; avoid English word-order calques.\n"
    "6. Cohesion and coherence: preserve logical relations, references, comparisons, cause-effect links, "
    "and paragraph-level flow.\n"
    "7. Academic fluency: use neutral textbook-style wording; avoid slang, casual pronouns, informal particles "
    "such as aku, kamu, nggak, sih, kok, dong, and deh, and Malay-specific forms when Indonesian has a "
    "standard equivalent.\n"
    "8. Structure preservation: preserve paragraph breaks, headings, numbering, bullet/list structure, "
    "examples, equations, LaTeX, citations, and code-like tokens exactly where possible.\n"
    "9. False-friend control: for example, translate 'policy' as 'kebijakan', not 'polisi', and choose "
    "context-appropriate renderings for words such as 'actual' and 'eventually'."
)


class TranslationGenerator:
    def __init__(
        self,
        client: SeaLionClient,
        model: str,
        max_generation_tokens: int,
    ) -> None:
        self.client = client
        self.model = model
        self.max_generation_tokens = max_generation_tokens

    @staticmethod
    def build_messages(
        source: str,
        stricter: bool = False,
    ) -> list[dict[str, str]]:
        strict_note = (
            "This is a retry after automated QA flagged the prior output. Be especially complete; "
            "do not omit sentences, headings, examples, formulas, or list items.\n"
            if stricter
            else ""
        )
        system_content = (
            "You are a professional English-to-Indonesian translator for educational and textbook-style "
            "materials. Produce faithful, fluent Indonesian that reads like standard written Indonesian, "
            "not a word-for-word English rendering.\n\n"
            f"{INDONESIAN_TRANSLATION_GUIDANCE}\n\n"
            "Return only the Indonesian translation. Do not add commentary, explanations, labels, or "
            "markdown fences."
        )
        user_content = (
            "Translate the following English educational text into standard formal Indonesian. "
            "Apply all Indonesian translation requirements from the system message.\n\n"
            f"{strict_note}"
            "<text>\n"
            f"{source}\n"
            "</text>"
        )
        return [
            {
                "role": "system",
                "content": system_content,
            },
            {"role": "user", "content": user_content},
        ]

    async def translate(
        self,
        source: str,
        stricter: bool = False,
    ) -> tuple[str, list[str]]:
        messages = self.build_messages(source, stricter=stricter)
        translated = await self.client.chat(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=self.max_generation_tokens,
        )
        return clean_model_output(translated), [source]
