import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time

# Page Configuration
st.set_page_config(
    page_title="IoT Anomaly Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main page styling */
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
    }
    
    /* Header styling */
    .title-container {
        background: linear-gradient(90deg, #1cb5e0 0%, #000851 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #000851 0%, #1cb5e0 100%);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #1cb5e0 0%, #000851 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Metrics styling */
    .css-1xarl3l {
        background: 90deg, #1cb5e0 0%, #000851 100%;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Card styling */
    .css-12w0qpk {
        background: 90deg, #1cb5e0 0%, #000851 100%;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1cb5e0 0%, #000851 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("""
    <div class="title-container">
        <h1>🔍 IoT Sensor Anomaly Detection</h1>
        <p>Real-time anomaly detection in IoT sensor data using LSTM Neural Networks</p>
    </div>
""", unsafe_allow_html=True)

# Define constants
SEQUENCE_LENGTH = 10  # Same as used in training

# Caching functions
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_model.keras")

@st.cache_data
def load_reference_data():
    df = pd.read_csv("dataset_final.csv")
    if 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'], unit='s')
        df.set_index('Time', inplace=True)
    return df

def create_sequences(data, sequence_length=SEQUENCE_LENGTH):
    X = []
    for i in range(len(data) - sequence_length + 1):
        X.append(data[i:i+sequence_length])
    return np.array(X)

# Load model and reference data
with st.spinner("Loading model and reference data..."):
    model = load_lstm_model()
    ref_data = load_reference_data()
    
    # Get model input shape for validation
    input_shape = model.input_shape
    st.write("Model expects input shape:", input_shape)

# Sidebar
with st.sidebar:
    st.markdown('<h2 style="color: #1cb5e0;">📊 Control Panel</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #1cb5e0;">Upload your IoT sensor data for analysis</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file:
        st.markdown('<p style="color: Green;">✅ File uploaded successfully!</p>', unsafe_allow_html=True)
        st.markdown("""
            <div style="color: white; margin-top: 2rem;">
                <h3>📝 Instructions</h3>
                <ol>
                    <li>Prepare your CSV file:
                        <ul style="margin-left: 20px; margin-top: 5px;">
                            <li>Required columns: Temperature, Humidity, Air Quality, Light, Loudness</li>
                            <li>Time column should be in Unix timestamp format</li>
                            <li>Data should be continuous time series</li>
                        </ul>
                    </li>
                    <li>Upload & Process:
                        <ul style="margin-left: 20px; margin-top: 5px;">
                            <li>Click 'Browse files' to upload your CSV</li>
                            <li>System will automatically normalize your data</li>
                            <li>Anomaly detection uses 10-step sequence patterns</li>
                        </ul>
                    </li>
                    <li>Analyze Results:
                        <ul style="margin-left: 20px; margin-top: 5px;">
                            <li>View time series visualization</li>
                            <li>Check reconstruction error distribution</li>
                            <li>Identify anomalous data points</li>
                        </ul>
                    </li>
                    <li>Export & Share:
                        <ul style="margin-left: 20px; margin-top: 5px;">
                            <li>Download results as CSV</li>
                            <li>Includes original data with anomaly labels</li>
                            <li>Perfect for further analysis</li>
                        </ul>
                    </li>
                </ol>
            </div>
            
            <div style="color: white; margin-top: 2rem;">
                <h3>💡 Tips</h3>
                <ul>
                    <li>Larger datasets provide better anomaly detection</li>
                    <li>Make sure your data is cleaned before uploading</li>
                    <li>All sensor values should be numeric</li>
                    <li>Check the data preview to confirm proper loading</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# Main content
if uploaded_file is not None:
    # Load and display data
    new_data = pd.read_csv(uploaded_file)
    
    # Process Time column if it exists
    if 'Time' in new_data.columns:
        new_data['Time'] = pd.to_datetime(new_data['Time'], unit='s')
        new_data.set_index('Time', inplace=True)
    
    # Display data info
    st.write("### Uploaded Data Preview")
    st.dataframe(new_data.head(), use_container_width=True)
    st.write("Uploaded data shape:", new_data.shape)
    st.write("Available columns:", new_data.columns.tolist())
    
    # Ensure columns match reference data
    ref_columns = ref_data.columns.tolist()
    if set(ref_columns) != set(new_data.columns):
        st.error(f"❌ Column mismatch! Model expects these columns: {ref_columns}")
        st.error(f"Your data has these columns: {new_data.columns.tolist()}")
        st.stop()
    
    # Reorder columns to match reference data
    new_data = new_data[ref_columns]
    
    # Data Processing
    with st.spinner("Processing data..."):
        # Normalize data
        scaler = MinMaxScaler()
        scaler.fit(ref_data)
        scaled_data = scaler.transform(new_data)
        
        st.write("Data shape after scaling:", scaled_data.shape)
        
        # Create sequences
        try:            # Create sequences using the constant
            sequences = create_sequences(scaled_data, SEQUENCE_LENGTH)
            st.write("Sequences shape:", sequences.shape)
            st.write("Expected shape: (num_samples, sequence_length, num_features)")
            st.write(f"Using sequence length: {SEQUENCE_LENGTH}")
            
            # Verify sequence dimensions match model input
            if sequences.shape[1:] != input_shape[1:]:
                st.error(f"❌ Input shape mismatch! Model expects {input_shape[1:]}, but got {sequences.shape[1:]}")
                st.stop()
            
            # Make predictions with proper batch size
            with st.spinner("Detecting anomalies..."):
                predictions = model.predict(sequences, batch_size=32, verbose=0)
                reconstruction_error = np.mean(np.square(sequences - predictions), axis=(1,2))
                
                # Calculate threshold and detect anomalies
                threshold = np.percentile(reconstruction_error, 95)
                anomalies = reconstruction_error > threshold
                
                # Add results to dataframe
                results_df = new_data.copy()
                results_df['Anomaly'] = False
                results_df.iloc[SEQUENCE_LENGTH-1:, -1] = anomalies
                
        except Exception as e:
            st.error(f"Error during processing: {str(e)}")
            st.write("Model input shape:", model.input_shape)
            st.write("Current data shape:", scaled_data.shape)
            st.stop()

    # Results Display
    st.markdown("### 🎯 Detection Results")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_points = len(results_df)
        anomaly_points = results_df['Anomaly'].sum()
        st.metric("Total Data Points", total_points)
        
    with col2:
        anomaly_percentage = (anomaly_points / total_points) * 100
        st.metric("Anomalies Detected", f"{anomaly_points} ({anomaly_percentage:.2f}%)")
        
    with col3:
        st.metric("Anomaly Threshold", f"{threshold:.6f}")

    # Visualization
    st.markdown("### 📊 Visualization")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Time Series View", "Detailed Analysis"])
    
    with tab1:
        # Time series plot with anomalies
        fig = go.Figure()
        
        # Add normal points
        fig.add_trace(go.Scatter(
            x=results_df.index[~results_df['Anomaly']],
            y=results_df[results_df.columns[0]][~results_df['Anomaly']],
            mode='lines+markers',
            name='Normal',
            line=dict(color='#1cb5e0')
        ))
        
        # Add anomaly points
        fig.add_trace(go.Scatter(
            x=results_df.index[results_df['Anomaly']],
            y=results_df[results_df.columns[0]][results_df['Anomaly']],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=10, symbol='x')
        ))
        
        fig.update_layout(
            title="Sensor Data with Detected Anomalies",
            xaxis_title="Time",
            yaxis_title="Sensor Value",
            template="plotly_white",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        # Reconstruction error distribution
        fig = px.histogram(
            reconstruction_error,
            title="Reconstruction Error Distribution",
            labels={'value': 'Reconstruction Error'},
            template="plotly_white"
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                     annotation_text="Anomaly Threshold")
        st.plotly_chart(fig, use_container_width=True)

    # Download results
    st.markdown("### 💾 Download Results")
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="anomaly_detection_results.csv",
        mime="text/csv"
    )

else:
    # Welcome message when no file is uploaded
    st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>👋 Welcome to IoT Anomaly Detection!</h2>
            <p>Upload your CSV file in the sidebar to get started.</p>
            <p>The system will analyze your sensor data and detect any anomalies using advanced LSTM neural networks.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Display sample visualizations or instructions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #1cb5e0 0%, #000851 100%); padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); min-height: 200px;">
                <h3 style='color: white;'><span style="font-size:1.5em;">📋</span> Requirements</h3>
                <ul style='color: #f5f7fa;'>
                    <li>CSV file with sensor data</li>
                    <li>Data format matching reference dataset</li>
                    <li>Continuous time series data</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
            <div style="background: linear-gradient(135deg, #000851 0%, #1cb5e0 100%); padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); min-height: 200px;">
                <h3 style='color: white;'><span style="font-size:1.5em;">🎯</span> Features</h3>
                <ul style='color: #f5f7fa;'>
                    <li>Real-time anomaly detection</li>
                    <li>Interactive visualizations</li>
                    <li>Downloadable results</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)