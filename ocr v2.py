import os
from fuzzywuzzy import fuzz, process
import pytesseract
import cv2
from PIL import Image, UnidentifiedImageError, ImageFilter, ImageOps
import numpy as np
import pandas as pd
SUBSTAT_RANGES = {
    "Crit. Rate": [6.3, 6.9, 7.5, 8.1, 8.7, 9.3, 9.9, 10.5],
    "Crit. DMG": [12.6, 13.8, 15.0, 16.2, 17.4, 18.6, 19.8, 21.0],
    "HP": [320, 360, 390, 430, 470, 510, 540, 580],
    "DEF%": [8.1, 9.0, 10.0, 10.9, 11.8, 12.8, 13.8, 14.7],
    "ATK%": [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6],
    "HP%": [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6],
    "Energy Regen": [6.8, 7.6, 8.4, 9.2, 10.0, 10.8, 11.6, 12.4],   
    "Basic Attack DMG Bonus": [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6],
    "Heavy Attack DMG Bonus": [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6],
    "Resonance Skill DMG Bonus": [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6],
    "Resonance Liberation DMG Bonus": [6.4, 7.1, 7.9, 8.6, 9.4, 10.1, 10.9, 11.6],
    "DEF": [40, 50, 60, 70],
    "ATK": [30, 40, 50, 60], 
}

# Tesseract PATH
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# List of specific columns to extract (no % columns)
COLUMNS = [
    "File Name", "ATK", "HP", "DEF", 
    "Energy Regen", "Crit. Rate", "Crit. DMG",
    "Basic Attack DMG Bonus", "Heavy Attack DMG Bonus",
    "Resonance Skill DMG Bonus", "Resonance Liberation DMG Bonus",
    "ATK%", "HP%", "DEF%"
]

# Function to find the closest match in a range (no threshold check)
# def find_closest(value, substat_list):
   #return min(substat_list, key=lambda x: abs(x - value))

#Closest looking value
def find_closest(value, substat_list):
    best_match = None
    best_score = -1  # Initialize with a low score

    value_str = str(value)  # Convert input to string

    for item in substat_list:
        item_str = str(item)  # Convert list item to string
        score = fuzz.ratio(value_str, item_str)

        if score > best_score:
            best_score = score
            best_match = item
    
    print("VALUE:",best_match)   
    print("----------------------------------------------------")
    return best_match

#Pre Process Images 
def preprocess_image(image_path):
    """Preprocesses the image for better OCR results and saves it."""
    try:
        image = Image.open(image_path)
    except UnidentifiedImageError:
        print(f"Unable to open image: {image_path}")
        return None

    # 1. Rescaling (Upscaling)
    width, height = image.size
    image = image.resize((width * 3, height * 3), Image.LANCZOS)

    # 2. Grayscale Conversion
    image = image.convert('L')

    #3. Denoising
    image = image.filter(ImageFilter.GaussianBlur(1))

    # 4. Invert Colors (After Grayscale, Before Binarization)
    image = ImageOps.invert(image)

    # 5. Binarization (Otsu's Thresholding)
  #image = np.array(image)  # Convert to NumPy array for OpenCV
    #_, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #image = Image.fromarray(image)  # Convert back to PIL Image

    """
    # --- Saving the preprocessed image (DEBUG) ---
   
    output_folder = "Preprocessed_5"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Extract filename from input path
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name) #Splitting the name and extention for the file.
    output_path = os.path.join(output_folder, f"{name}_preprocessed{ext}") #Adding _preprocessed to the name

    try:
        image.save(output_path)
        print(f"Preprocessed image saved to: {output_path}")
    except Exception as e:
        print(f"Error saving preprocessed image: {e}")
    """

    return image
# Function to process a single image and extract relevant data
def process_image(image_path):
    
    image = preprocess_image(image_path)
    
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(image=image, config=f'--psm {6}')
    print("RAW DATA=", text)
    
    # Parse the text to extract only the specified columns
    data = {col: None for col in COLUMNS[1:]}  # Initialize all columns with None (skip File Name for now)
    for line in text.split('\n'):
        if line.strip():  
            parts = line.split()
            if len(parts) >= 2:
                attribute = ' '.join(parts[:-1])  
                value = parts[-1]
                # Convert value to float if possible
                try:
                    value_float = float(value.replace('%', ''))  # Remove % if present and convert to float
                except ValueError:
                    continue  # Skip non-numeric values
                
                # Perform fuzzy matching to get the best match attribute
                best_match, _ = process.extractOne(attribute, COLUMNS[1:], scorer=fuzz.ratio)
                print("ATTRIBUTE:", best_match)
                # Always select the closest match for substats
                if value.endswith('%'):  # If the value ends with "% OTHERWISE go into other columns"
                    if "ATK" in best_match:
                        data["ATK%"] = find_closest(value_float, SUBSTAT_RANGES["ATK%"])
                    elif "HP" in best_match:
                        data["HP%"] = find_closest(value_float, SUBSTAT_RANGES["HP%"])
                    elif "DEF" in best_match:
                        data["DEF%"] = find_closest(value_float, SUBSTAT_RANGES["DEF%"])
                    elif "Crit. Rate" in best_match:
                        data["Crit. Rate"] = find_closest(value_float, SUBSTAT_RANGES["Crit. Rate"])
                    elif "Crit. DMG" in best_match:
                        data["Crit. DMG"] = find_closest(value_float, SUBSTAT_RANGES["Crit. DMG"])
                    elif "Energy Regen" in best_match:
                        data["Energy Regen"] = find_closest(value_float, SUBSTAT_RANGES["Energy Regen"])
                    elif "Basic Attack DMG Bonus" in best_match:
                        data["Basic Attack DMG Bonus"] = find_closest(value_float, SUBSTAT_RANGES["Basic Attack DMG Bonus"])
                    elif "Heavy Attack DMG Bonus" in best_match:
                        data["Heavy Attack DMG Bonus"] = find_closest(value_float, SUBSTAT_RANGES["Heavy Attack DMG Bonus"])
                    elif "Resonance Skill DMG Bonus" in best_match:
                        data["Resonance Skill DMG Bonus"] = find_closest(value_float, SUBSTAT_RANGES["Resonance Skill DMG Bonus"])
                    elif "Resonance Liberation DMG Bonus" in best_match:
                        data["Resonance Liberation DMG Bonus"] = find_closest(value_float, SUBSTAT_RANGES["Resonance Liberation DMG Bonus"])
                else: 
                    if "ATK" in best_match:
                        data["ATK"] = find_closest(value_float, SUBSTAT_RANGES["ATK"])
                    elif "HP" in best_match:
                        data["HP"] = find_closest(value_float, SUBSTAT_RANGES["HP"])
                    elif "DEF" in best_match:
                        data["DEF"] = find_closest(value_float, SUBSTAT_RANGES["DEF"])
                    else:
                        data[best_match] = value
    return data

# Main function to process multiple images
def process_images(image_folder):
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Error: The folder {image_folder} does not exist.")
    
    all_data = []
    
    # Loop through all images in the folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure the image type
            image_path = os.path.join(image_folder, filename)
            print(f"Processing image: {image_path}")
            image_data = process_image(image_path)
            if image_data:
                image_data["File Name"] = filename
                all_data.append(image_data)
    
    if not all_data:
        raise ValueError("No valid data extracted from the images.")
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(all_data, columns=COLUMNS)  # Ensure specified column order
    return df

if __name__ == "__main__":
    image_folder = "data"
    output_file = "sorted_attributes_2.11.xlsx"
    psm_value=4
    
    try:
        # Process images and get sorted data
        df = process_images(image_folder)
        
        # Save the data to an Excel file
        df.to_excel(output_file, index=False)
        print(f"Data has been sorted and saved to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
