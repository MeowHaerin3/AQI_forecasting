/**
 * Load a component into an element by ID or Class
 * @param {string} selector - The ID or class of the element (e.g., "#sidebar" or ".sidebar")
 * @param {string} file - The file path to load
 */
function loadComponent(selector, file) {
    fetch(file)
        .then(response => response.text())
        .then(data => {
            document.querySelector(selector).innerHTML = data;
            if (selector === ".main-content") {
                initializeDashboard(); // Ensure dashboard JS runs after content loads
            }
        })
        .catch(error => console.error(`Error loading ${file}:`, error));
}

// Ensure the script runs only after the DOM is fully loaded
document.addEventListener("DOMContentLoaded", function () {
    loadComponent(".sidebar", "components/sidebar.html");
    loadComponent(".main-content", "components/main-content.html");
});

/**
 * Function to initialize dashboard functionality after content is loaded
 */
function initializeDashboard() {
    // AQI Chart
    const ctx = document.getElementById("aqiChart")?.getContext("2d");
    if (ctx) {
        new Chart(ctx, {
            type: "line",
            data: {
                labels: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                datasets: [
                    {
                        label: "AQI Index",
                        data: [45, 52, 60, 58, 55, 62, 65],
                        borderColor: "#3498db",
                        tension: 0.4,
                        fill: true,
                        backgroundColor: "rgba(52, 152, 219, 0.2)",
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: "AQI Value",
                        },
                    },
                },
            },
        });
    }

    // Forecast Data
    const forecastData = [
        { day: "Monday", aqi: 45, status: "good" },
        { day: "Tuesday", aqi: 52, status: "moderate" },
        { day: "Wednesday", aqi: 60, status: "moderate" },
        { day: "Thursday", aqi: 58, status: "moderate" },
        { day: "Friday", aqi: 55, status: "moderate" },
        { day: "Saturday", aqi: 62, status: "unhealthy-sensitive" },
        { day: "Sunday", aqi: 65, status: "unhealthy-sensitive" },
    ];

    // Generate Forecast
    const forecastContainer = document.getElementById("forecast");
    if (forecastContainer) {
        forecastContainer.innerHTML = ""; // Clear existing content
        forecastData.forEach((day) => {
            forecastContainer.innerHTML += `
                <div class="forecast-day">
                    <h4>${day.day}</h4>
                    <div class="aqi-indicator ${day.status}">${day.aqi}</div>
                    <div>${day.status.replace("-", " ").toUpperCase()}</div>
                </div>
            `;
        });
    }
}
