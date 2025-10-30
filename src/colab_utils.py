import csv
from pathlib import Path
from google.colab import auth
import gspread
from google.auth import default


def colab_download_sheets(
    google_sheet_id: str,
    dimensions_sheet_name: str,
    questions_sheet_name: str,
    output_dir=".",
):
    auth.authenticate_user()
    creds, _ = default(
        scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"])
    gc = gspread.authorize(creds)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dimensions_csv_path = output_dir / "dimensions.csv"
    questions_csv_path = output_dir / "questions.csv"

    # Access the spreadsheet
    print(f"Connecting to Google Sheet with ID: {google_sheet_id}")
    try:
        sheet = gc.open_by_key(google_sheet_id)

        # Download dimensions sheet
        print(f"Downloading '{dimensions_sheet_name}' sheet...")
        dimensions_worksheet = sheet.worksheet(dimensions_sheet_name)
        dimensions_data = dimensions_worksheet.get_all_records()

        # Save dimensions to CSV
        if dimensions_data:
            with open(dimensions_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=dimensions_data[0].keys())
                writer.writeheader()
                writer.writerows(dimensions_data)
            print(f"✓ Saved dimensions to {dimensions_csv_path}")
        else:
            print(
                f"No data found in dimensions sheet '{dimensions_sheet_name}'")

        # Download questions sheet
        print(f"Downloading '{questions_sheet_name}' sheet...")
        questions_worksheet = sheet.worksheet(questions_sheet_name)
        questions_data = questions_worksheet.get_all_records()

        # Save questions to CSV
        if questions_data:
            with open(questions_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=questions_data[0].keys())
                writer.writeheader()
                writer.writerows(questions_data)
            print(f"✓ Saved questions to {questions_csv_path}")
        else:
            print(f"No data found in questions sheet '{questions_sheet_name}'")

        return str(dimensions_csv_path), str(questions_csv_path)

    except Exception as e:
        print(f"Error: {str(e)}")
        raise


# Inspect view webserver hosted out of colab
# from pyngrok import ngrok
# from google.colab import userdata
# import logging
# logger = logging.getLogger(__name__)
# try:
#     userdata.get('NGROK_AUTH_TOKEN')
# except userdata.SecretNotFoundError:
#     logger.warning(f"Get an auth token from http://ngrok.com/. Set it as NGROK_AUTH_TOKEN in the secrets panel")

# # Start a tunnel to the inspect view server
# ngrok.set_auth_token(userdata.get('NGROK_AUTH_TOKEN'))
# public_url = ngrok.connect(7575)
# print(f"Inspect View is available at: {public_url}")

# !inspect view start --host 0.0.0.0 --port 7575
