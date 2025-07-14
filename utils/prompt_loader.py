import yaml

class PromptLoader:

    @staticmethod
    def _replace_variables(data, variables):
        """
        Recursively replace variables in the dictionary
        """
        if isinstance(data, str):
            # Replace variables
            for key, value in variables.items():
                data = data.replace(f"${{{key}}}", str(value))
            return data
        elif isinstance(data, dict):
            # Recursively process dictionary
            return {k: PromptLoader._replace_variables(v, variables) for k, v in data.items()}
        elif isinstance(data, list):
            # Recursively process list
            return [PromptLoader._replace_variables(i, variables) for i in data]
        else:
            return data

    @staticmethod
    def get_prompt(prompt_path, prompt_name, prompt_variable=None):
        """
        Load YAML file and perform variable substitution
        """
        # Load YAML file
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_dict = yaml.safe_load(file)


        # Ensure the specified prompt_name exists
        if prompt_name not in prompt_dict:
            raise ValueError(f"No prompt found for the name: {prompt_name}")

        # Get the specified prompt block
        prompt_blocks = prompt_dict[prompt_name]

        # Perform variable substitution
        if prompt_variable:
            prompt_blocks = PromptLoader._replace_variables(
                prompt_blocks, 
                prompt_variable
            )

        return prompt_blocks

