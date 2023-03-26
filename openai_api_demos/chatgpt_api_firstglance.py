import openai

# need to get an api key somehow?

def exampleusercall(message):
    MODEL = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": message},
        ],
    )
    return response

if __name__ == '__main__':
    message = 'Why do they call it xbox 360?'
    print(exampleusercall(message))