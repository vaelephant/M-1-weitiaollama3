import PyPDF2
import re
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_extracted_text(text):
    """Clean and preprocess extracted text."""
    logging.info("Starting text cleaning")
    original_length = len(text)

    # Regular expression replacements
    patterns = {
        r'^(Introduction|Chapter \d+:|What is|Examples:|Chapter \d+)': '',
        r'ctitious': 'fictitious',
        r'ISBN[- ]13: \d{13}': '',
        r'ISBN[- ]10: \d{10}': '',
        r'Library of Congress Control Number : \d+': '',
        r'(\.|\?|\!)(\S)': r'\1 \2',
        r'All rights reserved|Copyright \d{4}': '',
        r'\n\s*\n': '\n',
        r'[^\x00-\x7F]+': ' ',
        r'\s{2,}': ' ',
        r'(\.)(\s)': r'\1\n'
    }

    for pattern, repl in patterns.items():
        text = re.sub(pattern, repl, text)

    # Remove all newlines and replace newlines only after periods
    text = text.replace('\n', ' ')
    cleaned_length = len(text)
    logging.info(f"Cleaned text from {original_length} to {cleaned_length} characters")

    return text

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    logging.info(f"Opening PDF file: {pdf_path}")
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + ' '
            logging.info(f"Extracted text from page {i+1}")

    return text

def process_all_pdfs(input_dir, output_dir):
    """Process all PDF files in the input directory and save cleaned texts."""
    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)
            extracted_text = extract_text_from_pdf(pdf_path)
            cleaned_text = clean_extracted_text(extracted_text)

            # Prepare output path
            base_filename = f"cleaned_{filename}"
            output_path = os.path.join(output_dir, base_filename)
            logging.info(f"Writing cleaned text to {output_path}")

            # Write cleaned text to output file
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write(cleaned_text)

            logging.info(f"Processed and saved {base_filename}")

def main():
    input_dir = '待清洗文件'
    output_dir = '已清洗文件'
    logging.info("Starting to process PDF files")

    # Make sure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process all PDFs in the directory
    process_all_pdfs(input_dir, output_dir)
    logging.info("All PDF files have been processed")

if __name__ == '__main__':
    main()
