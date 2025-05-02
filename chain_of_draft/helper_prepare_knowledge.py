import pandas as pd
from fpdf import FPDF
import argparse
import os
import requests
from datetime import datetime
from bs4 import BeautifulSoup
import re

def load_and_trim_csv(csv_file, meeting_date):
    """
    Load CSV file and retain rows after the given meeting date.
    """
    # Load CSV
    df = pd.read_csv(csv_file, parse_dates=['Meeting Date'])
    
    # Convert user input meeting date to datetime
    meeting_date = pd.to_datetime(meeting_date)

    # Filter dataframe
    trimmed_df = df[df['Meeting Date'] < meeting_date]
    
    if trimmed_df.empty:
        raise ValueError("No data available after the specified Meeting Date.")

    return trimmed_df

def extract_latest_values(df):
    """
    Extracts the most recent row for selected economic indicators.
    """

    # Ensure Meeting Date is in datetime format and sort in ascending order
    df = df.sort_values(by="Meeting Date", ascending=True)

    # Get the most recent row (last row after sorting)
    latest_row = df.iloc[-1]

    # Define key indicators
    key_indicators = {
        "Gross Domestic Product (GDP)": "Brave-Butters-Kelley Real Gross Domestic Product",
        "Consumer Price Index (CPI)": "Consumer Price Index",
        "MICH Personal Consumption Expenditures (PCE) Price Index": "Personal Consumption Expenditures Price Index",
        "Inflation Expectation": "Inflation Expectation",
        "3-Month US Treasury Bond Yield": "3 Month US Treasury Bond Yield",
        "Unemployment Rate": "Unemployment Rate",
        "M2 Money Supply": "M2 Money Supply",
        "6-Month US Treasury Bond Yield": "6 Month US Treasury Bond Yield",
        "Volatility Index (VIXCLS)": "Volatility Index (VIXCLS)"
    }

    # Columns that need a '%' appended
    percentage_columns = {
        "MICH Personal Consumption Expenditures (PCE) Price Index",
        "3-Month US Treasury Bond Yield",
        "Unemployment Rate",
        "6-Month US Treasury Bond Yield"
    }

    # Define binary indicators
    binary_indicators = {
        "Republican President": "Republican President",
        "Democrat President": "Democrat President",
        "Powell Fed Chair": "Powell Fed Chair",
        "Yellen Fed Chair": "Yellen Fed Chair",
        "Bernanke Fed Chair": "Bernanke Fed Chair",
        "Greenspan Fed Chair": "Greenspan Fed Chair",
        "Financial Crisis Indicator": "Financial Crisis Indicator"
    }

    # Extract key indicator values and add '%' for relevant columns
    extracted_data = {
        name: (f"{latest_row[col]}%" if name in percentage_columns else latest_row[col])
        for name, col in key_indicators.items() if col in df.columns
    }

    # Extract binary indicators as 'Yes' or omit if 0
    binary_data = {name: "Yes" for name, col in binary_indicators.items() if latest_row[col] == 1}

    return extracted_data, binary_data

def extract_probabilities_from_excel(file, meeting_date):
    """
    Extract probabilities of rate hike and cut from an Excel file based on the meeting_date.
    
    Args:
      excel_file (str): Path to the Excel file.
      meeting_date (str): The meeting date as a string in the format "YYYY-MM-DD".
    
    Returns:
      tuple: (prob_hike, prob_cut) as floats if a match is found.
    
    Raises:
      ValueError: If no matching date is found or if the date is before July 2016.
    """
    # Load the Excel file into a DataFrame
    df = pd.read_csv(file, parse_dates=['Release Date'])
    meeting_date = pd.to_datetime(meeting_date)

    # Search for a matching row
    match_row = df.loc[df['Release Date'] == meeting_date]

    if not match_row.empty:
        prob_hike = float(match_row['Prob Hike'].values[0])
        prob_cut = float(match_row['Prob Cut'].values[0])
        return prob_hike, prob_cut
    else:
        available_dates = df['Release Date'].sort_values().tolist()
        raise ValueError(
            f"No probabilities found for the provided meeting date {meeting_date.strftime('%Y-%m-%d')}. "
            f"Please double-check the date or ensure it is listed in your Excel file.\n"
            f"The earliest retrievable date is {available_dates[0].strftime('%Y-%m-%d')} and the latest date is {available_dates[-1].strftime('%Y-%m-%d')}."
        )

def generate_pdf(extracted_data, binary_data, output_pdf, prob_hike=None, prob_cut=None):
    """
    Generates a PDF report with extracted economic indicators.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    pdf.set_font("Arial", style="B", size=14)
    pdf.cell(200, 10, "Current Macroeconomic Indicators", ln=True, align="C")
    pdf.ln(10)

    # Add probability information if available
    if prob_hike is not None and prob_cut is not None:
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, 
            f"The Futures market is currently predicting a {prob_hike}% chance of a rate hike, and a {prob_cut}% chance of a rate cut.")
        pdf.ln(10)

    # Add extracted economic indicators
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "Below are the latest available values for some current macroeconomic indicators", ln=True, align="L")
    pdf.ln(5)

    for key, value in extracted_data.items():
        pdf.cell(200, 10, f"{key}: {value}", ln=True, align="L")
    
    pdf.ln(10)

    # Add binary indicators if present
    if binary_data:
        pdf.set_font("Arial", style="B", size=12)
        pdf.cell(200, 10, "Active Indicators:", ln=True, align="L")
        pdf.set_font("Arial", size=12)

        for key, value in binary_data.items():
            pdf.cell(200, 10, f"{key}: {value}", ln=True, align="L")
    



    pdf.output(output_pdf)
    print(f"PDF generated: {output_pdf}")

def prepare_pdf_report(csv_file, excel_file, meeting_date, output_pdf=None, output_trimmed_csv=None):
    """
    Prepares a PDF report and a trimmed CSV report for knowledge generation by:
      1. Loading and filtering the CSV data based on the meeting_date.
      2. Extracting the latest economic indicator values.
      3. Generating a PDF report saved to output_pdf.
      4. Saving the trimmed CSV data to output_trimmed_csv.
    
    If output file names are not provided, they will be generated dynamically based on meeting_date.
    
    Returns:
      tuple: (path to generated PDF, path to trimmed CSV)
    """
    # Load and trim CSV data
    trimmed_df = load_and_trim_csv(csv_file, meeting_date)

    prob_hike, prob_cut = extract_probabilities_from_excel(excel_file, meeting_date)

    
    # Dynamically generate filenames if not provided
    if output_pdf is None:
        output_pdf = f"knowledge/{meeting_date}_current_macro.pdf"
    if output_trimmed_csv is None:
        output_trimmed_csv = f"knowledge/{meeting_date}_historical_macro.csv"
    
    # Save the trimmed CSV file
    trimmed_df.to_csv(output_trimmed_csv, index=False)
    print(f"Trimmed CSV saved: {output_trimmed_csv}")
    
    # Generate PDF report
    extracted_data, binary_data = extract_latest_values(trimmed_df)
    generate_pdf(extracted_data, binary_data, output_pdf, prob_hike=prob_hike, prob_cut=prob_cut)
    
    return output_pdf, output_trimmed_csv

def download_latest_beige_book(meeting_date, output_pdf=None):
    """
    Downloads the most recent Beige Book PDF that was released on or before the given meeting_date.
    
    Process:
      1. Fetch the Beige Book archive page, which lists the "View by Year" options.
      2. Extract the year links (e.g. "2024", "2023", etc.) from the archive page.
      3. For each year page (e.g. https://www.federalreserve.gov/monetarypolicy/beigebook2020.htm for 2020):
         - Parse the page to find all PDF links that match the pattern "BeigeBook_YYYYMMDD.pdf".
         - Convert the YYYYMMDD from the filename into a date.
         - Only keep PDFs released on or before the provided meeting_date.
      4. From the candidates, choose the PDF with the most recent release date.
      5. Download that PDF and save it (using a dynamic filename if none is provided).
    
    Args:
      meeting_date (str): A string in "YYYY-MM-DD" format.
      output_pdf (str, optional): A filename for saving the downloaded PDF.
    
    Returns:
      str: The file path of the downloaded PDF.
    
    Raises:
      ValueError: If no appropriate Beige Book PDF is found or if a download error occurs.
    """
    # Convert meeting_date to a datetime object for comparisons.
    meeting_date = datetime.strptime(meeting_date, "%Y%m%d")
    formatted_date = meeting_date.strftime("%Y-%m-%d")
    meeting_dt = datetime.strptime(formatted_date, "%Y-%m-%d")
    
    # Step 1: Fetch the Beige Book archive page.
    archive_url = "https://www.federalreserve.gov/monetarypolicy/beige-book-archive.htm"
    archive_resp = requests.get(archive_url)
    if archive_resp.status_code != 200:
        raise ValueError(f"Error fetching archive page: {archive_url}")
    archive_soup = BeautifulSoup(archive_resp.content, "html.parser")
    
    # Step 2: Extract year option links.
    # The archive page displays clickable years (e.g., 2024, 2023, …) as the text of <a> tags.
    year_links = []
    for a in archive_soup.find_all("a", href=True):
        text = a.get_text(strip=True)
        if text.isdigit() and len(text) == 4:
            href = a['href']
            # Build full URL if the href is relative.
            if not href.startswith("http"):
                full_url = "https://www.federalreserve.gov/" + href.lstrip("/")
            else:
                full_url = href
            year_links.append((int(text), full_url))
    
    if not year_links:
        raise ValueError("No year option links found on the Beige Book archive page.")
    
    # Step 3: For each year page, find candidate PDF links.
    candidate_entries = []  # List of tuples: (release_date, pdf_url)
    for year, year_url in year_links:
        # Only consider years that are not after the meeting year.
        if meeting_dt.year < year:
            continue

        page_resp = requests.get(year_url)
        if page_resp.status_code != 200:
            continue
        page_soup = BeautifulSoup(page_resp.content, "html.parser")
        
        # Find all anchor tags whose href attribute matches the PDF pattern.
        # Example pattern: "BeigeBook_20200115.pdf"
        for a in page_soup.find_all("a", href=True):
            href = a['href']
            m = re.search(r"BeigeBook_(\d{8})\.pdf", href)
            if m:
                date_str = m.group(1)  # e.g., "20200115"
                try:
                    release_dt = datetime.strptime(date_str, "%Y%m%d")
                except ValueError:
                    continue
                # Only consider PDFs released on or before the meeting date.
                if release_dt <= meeting_dt:
                    if not href.startswith("http"):
                        pdf_url = "https://www.federalreserve.gov" + href
                    else:
                        pdf_url = href
                    candidate_entries.append((release_dt, pdf_url))
    
    if not candidate_entries:
        raise ValueError(f"No Beige Book PDFs found on or before meeting date {meeting_date}.")
    
    # Step 4: Choose the candidate with the latest release date.
    candidate_entries.sort(key=lambda x: x[0], reverse=True)
    selected_date, selected_pdf_url = candidate_entries[0]
    
    # Step 5: Download the selected PDF.
    if output_pdf is None:
        output_pdf = f"knowledge/BeigeBook_{selected_date.strftime('%Y%m%d')}.pdf"
    
    pdf_resp = requests.get(selected_pdf_url)
    if pdf_resp.status_code != 200:
        raise ValueError(f"Error downloading PDF from {selected_pdf_url} - Status code: {pdf_resp.status_code}")
    
    with open(output_pdf, "wb") as f:
        f.write(pdf_resp.content)
    
    print(f"Downloaded Beige Book PDF: {output_pdf} from {selected_pdf_url}")
    return output_pdf

def fetch_fomc_statement(meeting_date: str) -> str:
    """
    Scrape the FOMC public statement for a given meeting date.
    Expects meeting_date format to be 'YYYYMMDD'
    Returns the cleaned FOMC statement text.
    """
    url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{meeting_date}a.htm"
    print(f"Fetching FOMC statement from: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to retrieve the page: {e}")
        return ""

    soup = BeautifulSoup(response.text, 'html.parser')

    # Most FOMC statement paragraphs are wrapped in <p> inside a <div class="col-xs-12 col-sm-8 col-md-8">
    main_content = soup.find('div', class_='col-xs-12 col-sm-8 col-md-8')

    if not main_content:
        print("Could not find main FOMC statement content.")
        return ""

    # Extract all <p> tags but skip paragraphs with voting, implementation notes, or meta
    paragraphs = main_content.find_all('p')

    statement_text = []
    for p in paragraphs:
        text = p.get_text(strip=True)

        # Stop collecting if we've reached the voting or implementation note section
        if ("Voting for the monetary policy action" in text or
            "Implementation Note" in text or
            "Last Update" in text):
            break

        # Skip "Share" or preface line
        if text.startswith("Share") or "Federal Reserve issues FOMC statement" in text:
            continue

        statement_text.append(text)

    return "\n\n".join(statement_text)