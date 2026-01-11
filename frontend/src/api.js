import axios from 'axios';

const API_URL = 'http://localhost:8000';

export const analyzeImage = async (file, useDetector = true) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('use_detector', useDetector);

    try {
        const response = await axios.post(`${API_URL}/predict`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    } catch (error) {
        throw new Error(error.response?.data?.detail || 'Error communicating with the server');
    }
};
