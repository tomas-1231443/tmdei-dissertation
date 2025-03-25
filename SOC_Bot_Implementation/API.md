# SOC Bot API Documentation

This document describes the endpoints of the SOC Bot API, which is designed to process security alerts from QRadar SOAR and return predictions for Priority and Taxonomy. The API is built using FastAPI and is designed to integrate into your existing security operations environment.

## Endpoints

### 1. Health Check

- **Endpoint:** `GET /health`
- **Purpose:**  
  Verify that the API server is running and responsive.
- **Response:**  
  A JSON object indicating the server status.
  ```json
  {
      "status": "OK"
  }
  ```
- **Usage:**  
  Use this endpoint to perform a simple connectivity check before sending alerts.

---

### 2. Process Alert

- **Endpoint:** `POST /alerts`
- **Purpose:**  
  Receive and process an alert from QRadar SOAR, and return predictions for the alert's Priority and Taxonomy.
- **Request Format:**  
  The endpoint expects a JSON payload conforming to the following Pydantic model:
  ```json
  {
      "Issue_ID": "string",
      "Issue_Type": "string",
      "Status": "string",
      "Description": "string",
      "Custom_field_Alert_Technology": "string",
      "Custom_field_Incident_Description": "string",
      "Custom_field_Incident_Resolution_1": "string",
      "Custom_field_Request_Type": "string",
      "Custom_field_Source": "string",
      "Custom_field_Source_Alert_Rule_Name": "string",
      "Custom_field_Source_Alert_Id": "string",
      "Custom_field_Taxonomy": "string",
      "Priority": "string"
  }
  ```
  - **Note:**  
    - The **Description** field is preprocessed (cleaned and normalized) before vectorization.
    - The RF model, trained on historical data, uses the preprocessed **Description** to predict the appropriate **Priority** and **Taxonomy**.

- **Response Format:**  
  On success, the API returns a JSON object with the Issue ID and the predicted values:
  ```json
  {
      "Issue ID": "string",
      "Prediction": "Predicted Priority and Taxonomy (formatted as a string)"
  }
  ```
- **Workflow Overview:**  
  1. The raw alert JSON is received and converted into a dictionary.
  2. The alert is processed via a custom ingestion function.
  3. The processed alert is converted to a DataFrame and then vectorized.
  4. The pre-trained model predicts the Priority and Taxonomy.
  5. The results are logged and returned as a JSON response.

---

### 3. Feedback Endpoint (Planned for Future Implementation)

- **Endpoint:** `POST /feedback` *(Not Implemented)*
- **Purpose:**  
  In future versions, this endpoint will allow analysts to submit corrections to the model's predictions. The goal is to use this feedback to further train and fine-tune the ML model via reinforcement learning.
- **Expected Request Format:**  
  The JSON payload might include:
  ```json
  {
      "Issue_ID": "string",
      "Correct_Priority": "string",
      "Correct_Taxonomy": "string",
      "Description": "string"
  }
  ```
- **Intended Behavior:**  
  - The system will record the provided corrections.
  - Based on the feedback, the reinforcement learning module (yet to be implemented) will adjust or retrain the model to improve accuracy over time.
- **Note:**  
  This endpoint is currently **not implemented**. It is planned for future releases to enable a dynamic, learning feedback loop from analyst corrections.