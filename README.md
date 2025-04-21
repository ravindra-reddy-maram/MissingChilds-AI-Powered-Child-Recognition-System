# 🧠 MissingChilds – AI-Powered Child Recognition System

A **smart missing child identification platform** built with Django and Deep Learning. This project combines the power of facial recognition and modern web technologies to help identify and locate missing children with the aid of computer vision models like **VGG16** and **ResNet**.

## 📌 Project Overview

**MissingChilds** is designed to support law enforcement and the public in identifying missing children using facial recognition. Users can upload images and the system intelligently matches them against a trained dataset using deep learning models. It's a robust end-to-end solution integrating image processing, deep learning, and a web interface.

---

## 🛠️ Tech Stack

| Layer         | Technology                              |
| ------------- | --------------------------------------- |
| **Backend**   | Django, Python                          |
| **AI Models** | ResNet, VGG16, OpenCV (Haar Cascades)   |
| **Database**  | SQLite / Text-based (DB.txt)            |
| **Frontend**  | Django Templating, Bootstrap (optional) |
| **Others**    | Haarcascade XML, NumPy, Torch           |

---

## ✨ Features

- 🎯 **Face Detection** using Haar Cascades
- 🔍 **Face Recognition** via VGG16 and ResNet deep learning models
- 📂 **Image Upload & Matching** capability
- 📊 **Local DB Support** for face data management (DB.txt)
- ⚙️ **Test & Validate** using script-based inference (test.py)
- 🛡️ Scalable and modular **Django project structure**

---

## 📁 Project Structure

```
MissingChilds/
├── MissingChild/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
├── haarcascade_frontalface_default.xml  # Face detection model
├── resnet.py                            # ResNet classifier
├── vgg16.py                             # VGG16 classifier
├── test.py                              # Testing script
├── DB.txt                               # Face database entries
├── manage.py
├── SCREENS.docx                         # UI screenshots
└── MODIFIED_SCREENS.docx               # Updated UI
```

## 📃Documentation

🚀 Getting Started

1. Clone the Repository
   git clone [https://github.com/yourusername/MissingChilds.git](https://github.com/yourusername/MissingChilds.git)

cd MissingChilds

2. Set Up the Environment
   python -m venv venv

source venv/bin/activate  # For Windows: venv\Scripts\activate

pip install -r requirements.txt  # Create if missing

3. Run the Server
   python manage.py runserver

4. Test the Model
   python test.py

## 🤖 Model Insights

VGG16 and ResNet architectures are used for high-accuracy facial recognition.

Models are integrated using torch and OpenCV.

Preprocessing, encoding, and matching is handled in vgg16.py and resnet.py.

## 📌 Future Enhancements

Migrate database to PostgreSQL or MongoDB

Improve model training with larger datasets

Deploy on Heroku/AWS

Add user authentication and image approval flow

## 🙌 Contributing

Fork this repository.

Create your feature branch (git checkout -b feature/yourFeature).

Commit your changes (git commit -m 'Add some feature').

Push to the branch (git push origin feature/yourFeature).

Create a new Pull Request.

## 📄 License

This project is licensed under the MIT License.
