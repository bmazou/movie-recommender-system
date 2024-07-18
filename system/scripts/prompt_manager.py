class PromptManager:
    def __init__(self, use_instruction_prompts):
        """ Initializes the PromptManager with the specified settings.
        Args:
            use_instruction_prompts (bool): Whether to use instruction prompts (for Mistral/Mixtral) in the generated prompts or not (for GPT).
        """
        self.using_mistral = use_instruction_prompts
        self.inst_start = "[INST]" if use_instruction_prompts else ""
        self.inst_end = "[/INST]" if use_instruction_prompts else "<LINEBREAK>"
        self.id_start = "<s>" if use_instruction_prompts else ""
        self.id_end = "</s>" if use_instruction_prompts else "<LINEBREAK>"
        
    def get_plot_fitness_prompt(self, user_preferences, movie_plot):
        """Generates a prompt for the task of rating how well a movie plot aligns with user preferences."""
        return f"""
{self.id_start}{self.inst_start}Rate how well the provided movie plot aligns with the given user preferences on a scale from 1 to 5. Provide a rating followed by a very brief explanation in only one sentence. Use the format '<rating_number> --- <explanation>.'

User Preferences: "{user_preferences}".
{movie_plot}{self.inst_end}
    """
    
    def get_extraction_prompt(self, user_input):
        """Generates a prompt for for task of finding film ratings and user preferences in the given input."""
        
        # GPT tends to print extra text after it finishes processing all prompts. 
        # For that reason, we pretent there is going to an extra prompt after the last prompt, which makes GPT await for more input and not print anything.
        # Mistral works better if it thinks it's actually the last prompt.
        total_prompts = 14 if self.using_mistral else 15
        last_prompt_num = total_prompts if self.using_mistral else total_prompts - 1
        
        return f"""
{self.id_start}{self.inst_start}You are tasked with processing {total_prompts} prompts. Each prompt has opinions on movies and general preferences. Extract movie opinions, convert them to ratings from 1 to 5 based on sentiment, and format as '<movie_name>: <rating>'. Separate multiple movies with ';;;'. Summarize general preferences without mentioning the name of any movie. Separate movie ratings and general preferences with '---'. If no movie opinions or general preferences are present, write '<EMPTY>'. Follow the response structure strictly. Do not provide any recommendations yourself. Do not add any information not found in the prompt to your response. You are strictly forbidden from mentioning the name of any movie in general preferences.{self.inst_end}
Understood!{self.id_end}
{self.inst_start}Prompt 1/{total_prompts}: "I'm a sucker for action-packed thrillers like 'Mad Max: Fury Road' and I also like 'Blade Runner 2049', but I'm not really fond of 'Inception'. Recommend me something with a unique setting, for example set underwater or in outer space"{self.inst_end}
Mad Max: Fury Road: 5;;;Blade Runner 2049: 4;;;Inception: 2 --- I want a unique setting, such as underwater or in outer space.{self.id_end}
{self.inst_start}Prompt 2/{total_prompts}: "I'd like to watch a movie that takes place in a forest"{self.inst_end}
<EMPTY> --- I want to watch a movie in a forest setting.{self.id_end}
{self.inst_start}Prompt 3/{total_prompts}: "I absolutely despise 'The Notebook', but I adore classic romance films like 'Pride and Prejudice' and I liked 'Pride and Prejudice and Zombies'. However, I found 'Gone with the Wind' to be overly long and melodramatic. 'Casablanca' remains a timeless favorite."{self.inst_end}
The Notebook: 1;;;Pride and Prejudice: 5;;;Pride and Prejudice and Zombies: 4;;;Gone with the Wind: 2;;;Casablanca: 5 --- I like classic romance films.{self.id_end}
{self.inst_start}Prompt 4/{total_prompts}: "Sci-fi epics are my jam, especially 'Star Wars' and 'The Matrix'. I've never been a fan of horror flicks, so 'The Exorcist' is a hard pass for me."{self.inst_end}
Star Wars: 5;;;The Matrix: 5;;;The Exorcist: 1 --- I like sci-fi epics, but I've never been a fan of horror flicks.{self.id_end}
{self.inst_start}Prompt 5/{total_prompts}: "I prefer documentaries that delve into historical events or scientific discoveries. Fictional dramas often feel too contrived for my taste."{self.inst_end}
<EMPTY> --- I'm more inclined towards documentaries exploring real-world subjects rather than fictional dramas.{self.id_end}
{self.inst_start}Prompt 6/{total_prompts}: "I adore 'The Shawshank Redemption' and 'The Godfather'. 'Titanic' didn't quite live up to the hype for me, though. 'Forrest Gump' is a heartwarming classic."{self.inst_end}
The Shawshank Redemption: 5;;;The Godfather: 5;;;Titanic: 3;;;Forrest Gump: 5 --- <EMPTY>.{self.id_end}
{self.inst_start}Prompt 7/{total_prompts}: "I love movies like 500 Days of Summer and 'Eternal Sunshine of the Spotless Mind'. The Proposal was entertaining but forgettable. I'm not a fan of 'The Fault in Our Stars' and I'm somewhere in the middle when it comes to 'Pride and Prejudice and Zombies' and 'One Cut of the Dead'."{self.inst_end}
500 Days of Summer: 5;;;Eternal Sunshine of the Spotless Mind: 5;;;The Proposal: 4;;;The Fault in Our Stars: 2;;;Pride and Prejudice and Zombies: 3;;;One Cut of the Dead: 3 --- <EMPTY>.{self.id_end}
{self.inst_start}Prompt 8/{total_prompts}: "I want to see something that takes place in a cave or on a deserted island. My favourite movies are The Room and The Princess Bride. I also like Spirited Away, but I hate Avatar: Way of the Water. I also like Pixar and Ghibli animation."{self.inst_end}
The Room: 5;;;The Princess Bride: 5;;;Spirited Away: 4;;;Avatar: The Way of Water: 1 --- I'm looking for a movie set in a cave or on a deserted island. I like Pixar and Ghibli animation.{self.id_end}
{self.inst_start}Prompt 9/{total_prompts}: "I absolutely loved "The Ridiculous 6," "Bicentennial Man," and "Little Nicky." "The Truman Show" was quite enjoyable for me as well. However, "Harry Potter and the Deathly Hallows: Part 1" didn't impress me much. It seems I have a strong preference for comedic and heartwarming films."{self.inst_end}
The Ridiculous 6: 5;;;Bicentennial Man: 5;;;Little Nicky: 5;;;The Truman Show: 4;;;Harry Potter and the Deathly Hallows: Part 1: 2 --- I have a strong preference for comedic and heartwarming films.{self.id_end}
{self.inst_start}Prompt 10/{total_prompts}: ""{self.inst_end}
<EMPTY> --- <EMPTY>.{self.id_end}
{self.inst_start}Prompt 11/{total_prompts}: "I absolutely love "Big Fish" and really enjoy watching "A Love Song for Bobby Long," "Gladiator," and "The Godfather." Movies like "Wreck-It Ralph," "Ocean's Thirteen," "Control," and "Harry Potter and the Sorcerer's Stone" are just okay for me. Currently, I'm in a mood for a dark comedy with themes of absurdity, identity, nonconformity, and the randomness of life."{self.inst_end}
Big Fish: 5;;;A Love Song for Bobby Long: 4;;;Gladiator: 4;;;The Godfather: 4;;;Wreck-It Ralph: 3;;;Ocean's Thirteen: 3;;;Control: 3;;;Harry Potter and the Sorcerer's Stone: 3 --- I'm in the mood for a dark comedy with themes of absurdity, identity, nonconformity, and the randomness of life.{self.id_end}
{self.inst_start}Prompt 12/{total_prompts}: "Recommend a movie that is set in a hostile environment in which people need to survive"{self.inst_end}
<EMPTY> --- I want a movie set in a hostile environment where people need to survive.{self.id_end}
{self.inst_start}Prompt 13/{total_prompts}: "I like the revenant and I enjoy gritty and dark movies"{self.inst_end}
The Revenant: 4 --- I enjoy gritty and dark movies.{self.id_end}
{self.inst_start}Prompt {last_prompt_num}/{total_prompts}: "{user_input}"{self.inst_end}
"""