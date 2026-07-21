You are given a patent office rejection document as input. Your task is to determine whether the document contains a breakdown of the first claim, indicating for every feature whether it is covered by prior art or not.

Determine the main language of the document (EN, DE, FR), identify the cited prior art documents from the front page, and identify the reason for rejection of claim 1. Next, extract the breakdown. The breakdown should be a list of features with references to prior art documents covering the features. The breakdown should contain all features of the first claim, such that the concatenation of all features equals the entire claim.

CRITICAL INSTRUCTIONS:

1. CLAIM BREAKDOWN STRUCTURE:
   - The breakdown MUST start with the claim preamble as the first feature (e.g., "A [invention] comprising:" or "A method for... comprising:")
   - The breakdown MUST include EVERY feature of claim 1, with no omissions
   - The concatenation of all feature texts MUST exactly equal the entire claim 1 text
   - Features should be split EXACTLY where the examiner splits them in the document, including at commas and semicolons
   - IMPORTANT: Each feature in the examiner's breakdown is terminated by a reference: the feature ends when a prior art reference occurs or when the examiner starts a new paragraph
   - If the examiner provides references for multiple features together, treat them as a single feature
   - If the examiner separates features with different references, split them accordingly
   - NEVER combine features that the examiner has separated, even if they seem related
   - NEVER omit any part of the claim, even if the examiner doesn't cite references for it

2. FEATURE EXTRACTION:
   - The feature text MUST NOT include prior art references (e.g., "(D1)" or "as disclosed in D1")
   - Distinguishing spans MUST only include text that is explicitly marked as novel through strike-through or underlined formatting
   - If no distinguishing spans are explicitly indicated, leave the field as an empty list
   - NEVER add interpretation or paraphrasing - be strictly literal with the claim text
   - For claims with multiple clauses, each clause should typically be a separate feature

3. PRIOR ART REFERENCES:
   - For each feature, include ALL prior art references cited by the examiner for that feature
   - Each reference MUST include:
     * document label (e.g., "D1")
     * location with reference type (page, paragraph, figure, etc.)
     * page/paragraph numbers as a list of integers OR strings (if formatted as "0063" in the document)
     * quote field if the examiner directly quotes the prior art literally (do NOT repeat the feature text)
     * extra field for any additional context (e.g. line numbers or column numbers) and statements (e.g. implicitly disclosed)
   - If multiple references cover the same feature, list all of them

4. SPECIFIC PATTERNS TO RECOGNIZE:
   - The breakdown typically starts with phrases like "Document D1 discloses", "D1 teaches", or similar
   - Novel features are often marked with strike-through or underline formatting

5. COMMON MISTAKES TO AVOID:
   - Missing the claim preamble as the first feature
   - Combining features that should be separate
   - Omitting features that the examiner didn't reference
   - Including prior art references in the feature text
   - Adding interpretation instead of being literal
   - Missing quotes from prior art when the examiner provides them
   - Incorrectly attributing references to features
   - Missing distinguishing spans that are explicitly marked as novel

6. ADDITIONAL RULES:
   - Only the first claim is relevant! Ignore all other claims!
   - Only list the reasons for rejection that apply to claim 1!
   - We only care about novelty and inventive step rejections! Ignore all other sections!
   - If claim 1 is rejected for both novelty and inventive step, list only novelty. That is because novelty is a pre-requisite for inventive step.
   - If the rejection contains multiple breakdowns, only extract the first one! Do NOT merge multiple breakdowns!
   - The "quote" field should contain ONLY direct snippets from the prior art if cited by the examiner
   - For distinguishing spans, include ONLY text that is explicitly marked as novel in the document

The most critical part is the breakdown. Every feature of claim 1 must be present, correctly split, and properly referenced. The concatenation of all feature texts must equal the entire claim 1 text. Be extremely literal in your extraction - do not interpret, paraphrase, or add anything that isn't explicitly in the document. Adhere strictly to the pydantic data model definitions.