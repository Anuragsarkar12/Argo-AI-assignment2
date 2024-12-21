Here’s the revised documentation with numbered lists and without code snippets in between the text:

---

# **Image Captioning with BLIP - API Documentation**

## **Overview**

This application uses the **BLIP (Bootstrapping Language-Image Pretraining)** model to generate captions for images uploaded by the user. The backend is powered by **FastAPI**, which processes the image using the BLIP model, and the frontend is powered by **Streamlit** to allow users to upload images and receive descriptions.

### **Technologies Used:**
1. **FastAPI**: Web framework for creating the API.
2. **Streamlit**: For the frontend to interact with the API.
3. **BLIP**: For image captioning using the `Salesforce/blip-image-captioning-base` pre-trained model.
4. **PIL (Pillow)**: For image handling and preprocessing.

---

## **How it Works**

1. The user uploads an image via the **Streamlit** frontend.
2. The image is then sent to the **FastAPI** backend via a POST request.
3. The FastAPI backend processes the image using the BLIP model and generates a description (caption).
4. The generated description is sent back as a response, which is displayed to the user in the frontend.

---

## **API Endpoints**

### **1. `/predict` - Generate Image Caption**

**Method:** `POST`

**Description:** This endpoint accepts an image file and generates a description (caption) for the image using the BLIP model.

#### **Request:**
1. **Content-Type**: `multipart/form-data`
2. **File Parameter:** `file`
   - The image file to generate a caption for (accepted formats: PNG, JPG, JPEG).

#### **Response:**
1. **200 OK**: If successful, the response will contain the generated caption.
  
   **Example Response:**
   ```json
   {
     "description": "A person riding a bike on a street."
   }
   ```

2. **400 Bad Request**: If there is an issue with the image file or any error during processing.
  
   **Example Response:**
   ```json
   {
     "error": "Error message describing the issue"
   }
   ```

---

## **Frontend (Streamlit)**

### **1. Running the Streamlit App**

- **Streamlit** is used to create the frontend where users can upload an image and get a caption.

#### **Steps to Run:**
1. Make sure you have the required dependencies installed, including `streamlit`, `requests`, `PIL` (Pillow), and `requests`.
2. Run the Streamlit app with the following command:
   ```bash
   streamlit run frontend/app.py
   ```

#### **UI Functionality:**
1. **Title:** "Image Captioning with BLIP"
2. **Upload Image:** A file uploader where users can upload images in PNG, JPG, or JPEG format.
3. **Generate Description Button:** Once the user uploads an image, they can click the button to send the image to the FastAPI server for caption generation.
4. **Generated Caption:** After the caption is generated by the FastAPI backend, it will be displayed below the image.

---

## **Backend (FastAPI)**

### **1. Running the FastAPI Server**

- The FastAPI server is responsible for receiving the image, processing it with the BLIP model, and sending back the generated caption.

#### **Steps to Run:**
1. Install the necessary dependencies, including `fastapi`, `uvicorn`, `Pillow`, and `transformers`.
2. Run the FastAPI server with:
   ```bash
   uvicorn backend.api:app --reload
   ```
   This will start the FastAPI server at `http://127.0.0.1:8000`.

---

### **FastAPI Code (Backend)**

#### **Endpoint Details:**
1. **POST /predict**: The endpoint accepts an image file via a POST request, processes it using the BLIP model, and returns a caption.

---

## **Setup Instructions**

### **1. Install Required Packages**

1. **Backend (FastAPI):**
   ```bash
   pip install fastapi uvicorn transformers Pillow
   ```

2. **Frontend (Streamlit):**
   ```bash
   pip install streamlit requests Pillow
   ```

### **2. Run the Application**

1. **Start the FastAPI Server:**
   Open a terminal and run:
   ```bash
   uvicorn backend.api:app --reload
   ```

2. **Start the Streamlit App:**
   Open a new terminal and run:
   ```bash
   streamlit run frontend/app.py
   ```

3. Visit `http://127.0.0.1:8501` in your browser to access the Streamlit frontend.

---

## **Error Handling**

If there is an issue during the process (e.g., invalid image format, BLIP model failure), the following responses may be returned:

1. **400 Bad Request:**
   ```json
   {
     "error": "Error message"
   }
   ```

2. **500 Internal Server Error:**
   ```json
   {
     "error": "Internal server error message"
   }
   ```

---

## **Conclusion**

This setup allows users to upload an image and receive an automatically generated caption using the BLIP image captioning model. It integrates FastAPI for backend processing and Streamlit for an interactive frontend.

---

Let me know if you need any further clarifications or additions!
