def get_prompt(option, email_content):
    """
    Generates a specific prompt based on the user's selected option.
    
    Args:
        option (str): The selected option for the prompt.
        email_content (str): The content of the email to generate the prompt for.
        
    Returns:
        str: The generated prompt.
    """
    if option == "Overall Summary":
        return overall_summary_prompt(email_content)
    # Add more options as needed
    # elif option == "Another Option":
    #     return another_option_prompt(email_content)

def overall_summary_prompt(email_content):
    """
    Generates a prompt for the overall summary.
    
    Args:
        email_content (str): The content of the email to generate the prompt for.
        
    Returns:
        str: The generated overall summary prompt.
    """
    prompt = f"""
    Based on the email content provided:
    {email_content}
    
    Provide a comprehensive overall summary, including:
    - The total number of students mentioned (if applicable).
    - General strengths of the student(s) as described in the email.
    - Common areas for improvement for the student(s).
    - Any notable trends or additional observations mentioned in the feedback.
    """
    return prompt
