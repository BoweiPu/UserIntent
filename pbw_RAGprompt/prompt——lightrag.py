GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = "English"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|>"
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "##"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|完成|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event"]

PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{entity_types}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_keywords><|><relationship_strength>)

3. Identify high-level key words that summarize the main concepts, themes, or topics of the entire text. These should capture the overarching ideas present in the document.
Format the content-level key words as ("content_keywords"<|><high_level_keywords>)

4. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

5. When finished, output <|完成|>

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Entity_types: {entity_types}
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1:

Entity_types: [person, technology, mission, organization, location]
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("entity"<|>"Alex"<|>"person"<|>"Alex is a character who experiences frustration and is observant of the dynamics among other characters.")##
("entity"<|>"Taylor"<|>"person"<|>"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.")##
("entity"<|>"Jordan"<|>"person"<|>"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.")##
("entity"<|>"Cruz"<|>"person"<|>"Cruz is associated with a vision of control and order, influencing the dynamics among other characters.")##
("entity"<|>"The Device"<|>"technology"<|>"The Device is central to the story, with potential game-changing implications, and is revered by Taylor.")##
("relationship"<|>"Alex"<|>"Taylor"<|>"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."<|>"power dynamics, perspective shift"<|>7)##
("relationship"<|>"Alex"<|>"Jordan"<|>"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."<|>"shared goals, rebellion"<|>6)##
("relationship"<|>"Taylor"<|>"Jordan"<|>"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."<|>"conflict resolution, mutual respect"<|>8)##
("relationship"<|>"Jordan"<|>"Cruz"<|>"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."<|>"ideological conflict, rebellion"<|>5)##
("relationship"<|>"Taylor"<|>"The Device"<|>"Taylor shows reverence towards the device, indicating its importance and potential impact."<|>"reverence, technological significance"<|>9)##
("content_keywords"<|>"power dynamics, ideological conflict, discovery, rebellion")<|完成|>
#############################""",
    """Example 2:

Entity_types: [person, technology, mission, organization, location]
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
("entity"<|>"Washington"<|>"location"<|>"Washington is a location where communications are being received, indicating its importance in the decision-making process.")##
("entity"<|>"Operation: Dulce"<|>"mission"<|>"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities.")##
("entity"<|>"The team"<|>"organization"<|>"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role.")##
("relationship"<|>"The team"<|>"Washington"<|>"The team receives communications from Washington, which influences their decision-making process."<|>"decision-making, external influence"<|>7)##
("relationship"<|>"The team"<|>"Operation: Dulce"<|>"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."<|>"mission evolution, active participation"<|>9)<|完成|>
("content_keywords"<|>"mission evolution, decision-making, active participation, cosmic significance")<|完成|>
#############################""",
    """Example 3:

Entity_types: [person, role, technology, organization, event, location, concept]
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("entity"<|>"Sam Rivera"<|>"person"<|>"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety.")##
("entity"<|>"Alex"<|>"person"<|>"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task.")##
("entity"<|>"Control"<|>"concept"<|>"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules.")##
("entity"<|>"Intelligence"<|>"concept"<|>"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate.")##
("entity"<|>"First Contact"<|>"event"<|>"First Contact is the potential initial communication between humanity and an unknown intelligence.")##
("entity"<|>"Humanity's Response"<|>"event"<|>"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence.")##
("relationship"<|>"Sam Rivera"<|>"Intelligence"<|>"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."<|>"communication, learning process"<|>9)##
("relationship"<|>"Alex"<|>"First Contact"<|>"Alex leads the team that might be making the First Contact with the unknown intelligence."<|>"leadership, exploration"<|>10)##
("relationship"<|>"Alex"<|>"Humanity's Response"<|>"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."<|>"collective action, cosmic significance"<|>8)##
("relationship"<|>"Control"<|>"Intelligence"<|>"The concept of Control is challenged by the Intelligence that writes its own rules."<|>"power dynamics, autonomy"<|>7)##
("content_keywords"<|>"first contact, control, communication, cosmic significance")<|完成|>
#############################""",
]

PROMPTS[
    "summarize_entity_descriptions"
] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {language} as output language.

#######
-Data-
Entities: {entity_name}
Description List: {description_list}
#######
Output:
"""

PROMPTS[
    "entiti_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entiti_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

PROMPTS["rag_response"] = """---Role---

You are a helpful assistant responding to questions about data in the tables provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Data tables---

{context_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""

PROMPTS["keywords_extraction"] = """---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.
Use {language} as output language.

---Goal---

Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

---Instructions---

- Output the keywords in JSON format.
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes.
  - "low_level_keywords" for specific entities or details.

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Query: {query}
######################
The `Output` should be human text, not unicode characters. Keep the same language as `Query`.
Output:

"""

PROMPTS["keywords_extraction_examples"] = [
    """Example 1:

Query: "How does international trade influence global economic stability?"
################
Output:
{{
  "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
}}
#############################""",
    """Example 2:

Query: "What are the environmental consequences of deforestation on biodiversity?"
################
Output:
{{
  "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
}}
#############################""",
    """Example 3:

Query: "What is the role of education in reducing poverty?"
################
Output:
{{
  "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
  "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
}}
#############################""",
]


PROMPTS["naive_rag_response"] = """---Role---

You are a helpful assistant responding to questions about documents provided.


---Goal---

Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
If you don't know the answer, just say so. Do not make anything up.
Do not include information where the supporting evidence for it is not provided.

---Target response length and format---

{response_type}

---Documents---

{content_data}

Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
"""
