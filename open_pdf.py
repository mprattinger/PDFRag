import PyPDF2

# Path to the PDF file
pdf_path = "Regeln.pdf"

# Open the PDF file
with open(pdf_path, "rb") as file:
    reader = PyPDF2.PdfReader(file)
    # Print number of pages
    print(f"Number of pages: {len(reader.pages)}")
    # Print text from the first page
    if reader.pages:
        first_page = reader.pages[0]
        print(first_page.extract_text())
    else:
        print("No pages found in the PDF.")
