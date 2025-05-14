# **SOC Bot API Specification**

**Version:** 1.0
**Base URL:** `http://<HOST>:8000`

---

## Overview

The SOC Bot API exposes endpoints for classifying incoming security alerts and ingesting analyst feedback for adaptive retraining. It leverages a hybrid Random Forest and Reinforcement Learning model pipeline.

This document specifies the endpoints required for integration into automated SOAR playbooks, as well as auxiliary endpoints that support monitoring and system health checks.

---

## Endpoints Summary

| Endpoint           | Method | Purpose                                    |
| ------------------ | ------ | ------------------------------------------ |
| `/alerts/final`    | POST   | Returns final classification for an alert. |
| `/alerts/feedback` | POST   | Submits feedback for RL model retraining.  |
| `/health`          | GET    | Returns API availability status.           |
| `/queue_length`    | GET    | Returns number of pending Celery tasks.    |

---

## `/alerts/final`

### **Description:**

Processes an alert and returns its final classification using either the Random Forest (RF) model or a Reinforcement Learning (RL) adjusted prediction, depending on confidence.

### **Method:** `POST`

### **Content-Type:** `application/json`

### **Request Body:**

```json
{
  "description": "<string>",
  "rule_name": "<string>"
}
```

| Field       | Type     | Description                                |
| ----------- | -------- | ------------------------------------------ |
| description | `string` | Full textual description of the alert.     |
| rule\_name  | `string` | Name of the rule that triggered the alert. |

---

### **Response:**

```json
{
  "Priority": "<string>",
  "Taxonomy": "<string>",
  "Is_FP": <boolean>,
  "RL_conf": <float>,
  "RF_conf": <float>,
  "Used": "RF" | "RL"
}
```

| Field    | Type      | Description                                                    |
| -------- | --------- | -------------------------------------------------------------- |
| Priority | `string`  | Final predicted alert priority.                                |
| Taxonomy | `string`  | Final predicted taxonomy (e.g., type of threat or context).    |
| Is\_FP   | `boolean` | Whether the alert is predicted as a false positive.            |
| RL\_conf | `float`   | Confidence score from the RL model.                            |
| RF\_conf | `float`   | Average confidence score from the RF model.                    |
| Used     | `string`  | Indicates which model's prediction was used: `"RF"` or `"RL"`. |

---

## `/alerts/feedback`

### **Description:**

Accepts analyst feedback for a previously classified alert and schedules a background RL retraining task if criteria are met.

### **Method:** `POST`

### **Content-Type:** `application/json`

### **Request Body:**

```json
{
  "description": "<string>",
  "rule_name": "<string>",
  "correct_priority": "<string>",
  "correct_taxonomy": "<string>",
  "resolution": "<string>"
}
```

| Field             | Type     | Description                                                               |
| ----------------- | -------- | ------------------------------------------------------------------------- |
| description       | `string` | Alert description provided to the analyst.                                |
| rule\_name        | `string` | Rule that triggered the alert.                                            |
| correct\_priority | `string` | Analyst-provided ground-truth priority (e.g., P1, P2, P3, P4).            |
| correct\_taxonomy | `string` | Analyst-provided ground-truth taxonomy.                                   |
| resolution        | `string` | Resolution status (e.g., "True Positive", "Duplicate", "Not Applicable"). |

### **Behavior:**

* Feedback is **ignored** and no retraining is triggered if:

  * `correct_priority == "P4"`
  * `correct_taxonomy == "other"`
  * `resolution` is one of: `"Duplicate"`, `"Done"`, `"Declined"`, `"Not Applicable/Not confirmed"`

---

### **Response:**

```json
{
  "status": "Task queued",
  "task_id": "<celery_task_id>"
}
```

or if skipped:

```json
{
  "status": "Training skipped due to feedback conditions."
}
```

---

## `/health`

### **Description:**

Verifies that the SOC Bot API service is alive and reachable.

### **Method:** `GET`

### **Response:**

```json
{
  "status": "OK"
}
```

---

## `/queue_length`

### **Description:**

Returns the current number of tasks waiting in the Celery queue.

### **Method:** `GET`

### **Response:**

```json
{
  "queue_length": <integer>
}
```

---

## Notes

* The `/alerts/final` endpoint is the primary inference interface.
* The `/alerts/feedback` endpoint facilitates continuous learning.