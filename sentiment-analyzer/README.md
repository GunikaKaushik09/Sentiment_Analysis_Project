# Sentiment Analyzer

This project is a web application that performs sentiment analysis on user-provided text using a machine learning model. The application is built with Flask and utilizes a sentiment analysis pipeline from the Hugging Face Transformers library.

## Project Structure

```
sentiment-analyzer
├── app.py                  # Main application script
├── templates               # Directory for HTML templates
│   └── index.html         # HTML template for user interface
├── static                  # Directory for static files
│   ├── css
│   │   └── style.css      # CSS styles for the web application
│   └── js
│       └── script.js      # JavaScript code for interactivity
├── sentiment_analysis.csv   # Dataset for training/testing the model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/sentiment-analyzer.git
   cd sentiment-analyzer
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000/`.

3. Enter text in the input field and submit to see the sentiment analysis results.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.