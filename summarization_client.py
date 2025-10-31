import requests
import json

# API endpoint
url = 'http://127.0.0.1:5000/summarize'
headers = {'Content-Type': 'application/json'}

# Example text
text = """
Python is one of the most popular programming languages in the world today. It is known for its readability and simplicity, making it an excellent choice for beginners. Python was developed by Guido van Rossum and first released in 1991. Over the years, it has become a dominant language in fields like web development, data science, machine learning, and automation.

Python's syntax is easy to learn and intuitive, which allows programmers to write less code to accomplish more. This makes it great for rapid prototyping and development. Python's simplicity also translates into a large and active community. There are many resources available online, including tutorials, documentation, and forums where people can seek help.

In addition to its ease of use, Python has a rich ecosystem of libraries and frameworks. Libraries like NumPy and pandas make data manipulation and analysis straightforward. For machine learning, TensorFlow, PyTorch, and scikit-learn are widely used libraries that help developers build complex models with ease. Django and Flask are two popular web frameworks that make it easy to develop web applications quickly.

Python is highly versatile, running on various platforms, including Windows, macOS, and Linux. It also has a number of powerful tools for automating tasks, like web scraping with BeautifulSoup or Selenium, or task scheduling with Celery. Python's versatility has made it a go-to tool for many software developers, analysts, and engineers.

Despite its popularity, Python is not without its limitations. It is slower than compiled languages like C++ and Java, making it less ideal for applications that require extreme performance. However, Python can still be used effectively in many high-performance applications by integrating with faster languages or using optimizations like NumPy's array processing.

The future of Python looks bright. It continues to grow in popularity due to its active development and large community. Python is constantly evolving, with regular updates and new features being added to make it even more powerful and user-friendly. As more industries adopt data science and machine learning, Python is expected to remain a leading language in the tech industry for years to come.
"""

# Prepare the request data
data = {
    'text': text,
    'max_length': 1024,
    'min_length': 64
}

try:
    print("Sending request to summarization API")
    
    # Send POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Check if request was successful
    if response.status_code == 200:
        result = response.json()
        print("\nSUMMARY:")
        print(result['summary'])
        
    else:
        print(f"Error {response.status_code}:", response.text)

    
except Exception as e:
    print({str(e)})
