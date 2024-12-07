from flask import Flask, request, render_template, redirect, url_for, session
import os
import threading

app = Flask(_name_)  # Corrected app instantiation
app.secret_key = 'your_secret_key'  # Necessary for session management

# Function to run the Streamlit app in a separate thread
def run_streamlit():
    os.system("streamlit run main.py --server.port 8502")  # Run Streamlit on port 8502

@app.route('/')
def home():
    return redirect(url_for('login'))  # Redirect to the login page

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Dummy authentication logic (replace with actual logic)
        if email == 'gargpunit2004@gmail.com' and password == 'password123':
            session['logged_in'] = True

            # Start Streamlit app in a new thread if not already running
            if not any(thread.name == "StreamlitThread" for thread in threading.enumerate()):
                thread = threading.Thread(target=run_streamlit, name="StreamlitThread")
                thread.daemon = True
                thread.start()

            # Redirect to Streamlit app
            return redirect("http://localhost:8502/")  # Redirect to Streamlit
        else:
            return "Login failed. Please try again.", 401

    return render_template('login.html')  # Serve login page for GET request

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

if _name_ == '_main_':
    app.run(debug=True, port=5000)  # Flask runs on port 5000