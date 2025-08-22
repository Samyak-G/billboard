import React, { useState, useEffect, useRef } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Image, Alert } from 'react-native';
import { Camera } from 'expo-camera';
import * as Location from 'expo-location';

export default function App() {
  const [hasCameraPermission, setHasCameraPermission] = useState(null);
  const [hasLocationPermission, setHasLocationPermission] = useState(null);
  const [location, setLocation] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      // Request camera permissions
      const cameraStatus = await Camera.requestCameraPermissionsAsync();
      setHasCameraPermission(cameraStatus.status === 'granted');

      // Request location permissions
      const locationStatus = await Location.requestForegroundPermissionsAsync();
      setHasLocationPermission(locationStatus.status === 'granted');

      if (locationStatus.status !== 'granted') {
        Alert.alert('Permission Denied', 'Location access is required to geotag reports.');
        return;
      }

      // Get current location
      try {
        let currentLocation = await Location.getCurrentPositionAsync({});
        setLocation(currentLocation);
      } catch (error) {
        Alert.alert('Location Error', 'Failed to fetch location. Please ensure GPS is enabled.');
      }
    })();
  }, []);

  const takePicture = async () => {
    if (cameraRef.current) {
      const options = { quality: 0.5, base64: true };
      const data = await cameraRef.current.takePictureAsync(options);
      setCapturedImage(data.uri);
    }
  };

  const uploadReport = async () => {
    if (!capturedImage || !location) {
      Alert.alert('Missing Data', 'Cannot upload without an image and location.');
      return;
    }

    // --- IMPORTANT ---
    // Replace with your computer's local IP address to connect from the Expo Go app.
    // Your phone and computer must be on the same Wi-Fi network.
    const API_URL = 'http://192.168.1.100:8000/reports'; 
    
    const formData = new FormData();
    
    // The backend expects a file, so we need to format the URI correctly
    const uriParts = capturedImage.split('.');
    const fileType = uriParts[uriParts.length - 1];

    formData.append('file', {
      uri: capturedImage,
      name: `report.${fileType}`,
      type: `image/${fileType}`,
    });

    formData.append('lat', location.coords.latitude);
    formData.append('lon', location.coords.longitude);
    // Optional fields can be added here if you have inputs for them
    // formData.append('notes', 'A test note from the app');

    try {
      console.log('Uploading to:', API_URL);
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      const jsonResponse = await response.json();

      if (response.ok) {
        Alert.alert(
          'Upload Successful',
          `Report created with ID: ${jsonResponse.id}`,
          [{ text: 'OK', onPress: () => setCapturedImage(null) }]
        );
      } else {
        // Handle server-side errors (e.g., validation)
        throw new Error(jsonResponse.detail || 'An unknown error occurred');
      }
    } catch (error) {
      console.error('Upload failed:', error);
      Alert.alert(
        'Upload Failed',
        `Could not connect to the server or an error occurred: ${error.message}. Make sure the API_URL is correct and the server is running.`
      );
    }
  };

  if (hasCameraPermission === null || hasLocationPermission === null) {
    return <View />;
  }
  if (hasCameraPermission === false) {
    return <Text>No access to camera</Text>;
  }
  if (hasLocationPermission === false) {
    return <Text>No access to location</Text>;
  }

  return (
    <View style={styles.container}>
      {capturedImage ? (
        <View style={styles.previewContainer}>
          <Image source={{ uri: capturedImage }} style={styles.previewImage} />
          <View style={styles.previewButtons}>
            <TouchableOpacity style={styles.button} onPress={() => setCapturedImage(null)}>
              <Text style={styles.buttonText}>Retake</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.button} onPress={uploadReport}>
              <Text style={styles.buttonText}>Upload Report</Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <Camera style={styles.camera} type={Camera.Constants.Type.back} ref={cameraRef}>
          <View style={styles.buttonContainer}>
            <TouchableOpacity style={styles.captureButton} onPress={takePicture} />
          </View>
        </Camera>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'flex-end',
    marginBottom: 30,
  },
  captureButton: {
    width: 70,
    height: 70,
    borderRadius: 35,
    backgroundColor: '#fff',
    borderWidth: 5,
    borderColor: '#ccc',
  },
  previewContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  previewImage: {
    width: '100%',
    height: '80%',
    resizeMode: 'contain',
  },
  previewButtons: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    width: '100%',
    padding: 20,
  },
  button: {
    backgroundColor: '#007BFF',
    padding: 15,
    borderRadius: 5,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
  },
});