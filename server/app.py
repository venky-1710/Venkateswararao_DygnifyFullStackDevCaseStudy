from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import httpx
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
from scipy import stats
import pandas as pd
from scipy.stats import mannwhitneyu
import os
from dotenv import load_dotenv
import logging
import uvicorn
from starlette.websockets import WebSocketState


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Climate Hazard Analysis API",
    description="API for analyzing historical weather data and climate hazards",
    version="1.0.0"
)

# Update CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB connection
MONGODB_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = "climate_analysis"
client = AsyncIOMotorClient(MONGODB_URL)
database = client[DATABASE_NAME]

# Collections
weather_collection = database.weather_data
analysis_collection = database.analysis_results
regions_collection = database.regions

# Pydantic models
class RegionInput(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    name: Optional[str] = None

class TimeRange(BaseModel):
    start_year: int = Field(..., ge=1980, le=2024)
    end_year: int = Field(..., ge=1980, le=2024)

class AnalysisRequest(BaseModel):
    region: RegionInput
    hazard_type: str = Field(..., pattern="^(heatwave|drought|heavy_rain)$")
    time_range: TimeRange

class WeatherData(BaseModel):
    date: datetime
    temperature_max: float
    temperature_min: float
    precipitation: float
    humidity: float
    region_id: str

class HazardEvent(BaseModel):
    start_date: datetime
    end_date: datetime
    duration: int
    intensity: float
    hazard_type: str

class TrendData(BaseModel):
    year: int
    frequency: int
    intensity: float
    duration: float

class AnalysisResult(BaseModel):
    region: RegionInput
    hazard_type: str
    time_range: TimeRange
    trends: List[TrendData]
    summary: Dict[str, Any]
    created_at: datetime

# NOAA NCEI API configuration
NOAA_API_BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2"
NOAA_API_TOKEN = os.getenv("NOAA_API_TOKEN", "wBIbZXdwTsGRMVWdMInHXACOvAxdrEDj")

class WeatherDataService:
    def __init__(self):
        self.headers = {
            "token": NOAA_API_TOKEN
        }
    
    async def fetch_historical_data(self, lat: float, lon: float, start_date: str, end_date: str) -> List[Dict]:
        """Fetch historical weather data from NOAA API"""
        try:
            async with httpx.AsyncClient() as client:
                # In a real implementation, you would call the NOAA API
                # For demo purposes, we'll generate mock data
                return await self._generate_mock_data(lat, lon, start_date, end_date)
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return await self._generate_mock_data(lat, lon, start_date, end_date)
    
    async def _generate_mock_data(self, lat: float, lon: float, start_date: str, end_date: str) -> List[Dict]:
        """Generate mock weather data for demonstration"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        data = []
        current_date = start
        
        # Base temperature varies by latitude
        base_temp = 70 - (abs(lat - 40) * 0.5)
        
        while current_date <= end:
            # Simulate seasonal temperature variation
            day_of_year = current_date.timetuple().tm_yday
            seasonal_factor = 20 * np.sin(2 * np.pi * day_of_year / 365.25)
            
            # Add some randomness and long-term warming trend
            year_offset = (current_date.year - 2000) * 0.02  # Small warming trend
            random_factor = np.random.normal(0, 5)
            
            temp_max = base_temp + seasonal_factor + year_offset + random_factor
            temp_min = temp_max - np.random.uniform(10, 20)
            
            # Precipitation
            precip = max(0, np.random.exponential(0.1))
            
            data.append({
                "date": current_date.isoformat(),
                "temperature_max": round(temp_max, 1),
                "temperature_min": round(temp_min, 1),
                "precipitation": round(precip, 2),
                "humidity": round(np.random.uniform(30, 90), 1)
            })
            
            current_date += timedelta(days=1)
        
        return data

class HazardAnalyzer:
    @staticmethod
    def detect_heatwaves(weather_data: List[Dict], threshold_percentile: float = 95) -> List[HazardEvent]:
        """Detect heatwave events: 3+ consecutive days above 95th percentile"""
        if not weather_data:
            return []
        
        # Calculate threshold temperature
        temps = [d["temperature_max"] for d in weather_data]
        threshold = np.percentile(temps, threshold_percentile)
        
        events = []
        current_event_start = None
        consecutive_days = 0
        
        for i, day in enumerate(weather_data):
            if day["temperature_max"] > threshold:
                if current_event_start is None:
                    current_event_start = i
                consecutive_days += 1
            else:
                if current_event_start is not None:
                    if consecutive_days >= 3:  # Heatwave threshold
                        start_date = datetime.fromisoformat(weather_data[current_event_start]["date"])
                        end_date = datetime.fromisoformat(weather_data[i-1]["date"])
                        
                        # Calculate intensity (average temperature above threshold)
                        event_temps = [weather_data[j]["temperature_max"] for j in range(current_event_start, i)]
                        intensity = np.mean([t - threshold for t in event_temps])
                        
                        events.append(HazardEvent(
                            start_date=start_date,
                            end_date=end_date,
                            duration=consecutive_days,
                            intensity=round(intensity, 2),
                            hazard_type="heatwave"
                        ))
                    
                    current_event_start = None
                    consecutive_days = 0
        
        # Check if the last event is a heatwave
        if current_event_start is not None and consecutive_days >= 3:
            start_date = datetime.fromisoformat(weather_data[current_event_start]["date"])
            end_date = datetime.fromisoformat(weather_data[-1]["date"])
            
            # Calculate intensity (average temperature above threshold)
            event_temps = [weather_data[j]["temperature_max"] for j in range(current_event_start, len(weather_data))]
            intensity = np.mean([t - threshold for t in event_temps])
            
            events.append(HazardEvent(
                start_date=start_date,
                end_date=end_date,
                duration=consecutive_days,
                intensity=round(intensity, 2),
                hazard_type="heatwave"
            ))
        
        return events
    
    @staticmethod
    def detect_heavy_rain(weather_data: List[Dict], threshold_percentile: float = 95) -> List[HazardEvent]:
        """Detect heavy rainfall events: precipitation above 95th percentile"""
        if not weather_data:
            return []
        
        # Calculate threshold
        precips = [d["precipitation"] for d in weather_data]
        if not precips:
            return []
        
        threshold = np.percentile(precips, threshold_percentile)
        
        events = []
        for i, day in enumerate(weather_data):
            if day["precipitation"] > threshold:
                start_date = datetime.fromisoformat(day["date"])
                
                events.append(HazardEvent(
                    start_date=start_date,
                    end_date=start_date,
                    duration=1,
                    intensity=round(day["precipitation"] - threshold, 2),
                    hazard_type="heavy_rain"
                ))
        
        return events

    @staticmethod
    def detect_droughts(weather_data: List[Dict], threshold_days: int = 30) -> List[HazardEvent]:
        """Detect drought events: 30+ consecutive days with minimal precipitation"""
        if not weather_data:
            return []
        
        # Calculate threshold (20th percentile of non-zero precipitation)
        precips = [d["precipitation"] for d in weather_data if d["precipitation"] > 0]
        if not precips:
            return []
        
        threshold = np.percentile(precips, 20)
        
        events = []
        current_event_start = None
        consecutive_days = 0
        
        for i, day in enumerate(weather_data):
            if day["precipitation"] <= threshold:
                if current_event_start is None:
                    current_event_start = i
                consecutive_days += 1
            else:
                if current_event_start is not None:
                    if consecutive_days >= threshold_days:  # Drought threshold
                        start_date = datetime.fromisoformat(weather_data[current_event_start]["date"])
                        end_date = datetime.fromisoformat(weather_data[i-1]["date"])
                        
                        # Calculate intensity (precipitation deficit)
                        event_precips = [weather_data[j]["precipitation"] for j in range(current_event_start, i)]
                        intensity = threshold - np.mean(event_precips)
                        
                        events.append(HazardEvent(
                            start_date=start_date,
                            end_date=end_date,
                            duration=consecutive_days,
                            intensity=round(max(0, intensity), 2),
                            hazard_type="drought"
                        ))
                    
                    current_event_start = None
                    consecutive_days = 0
        
        # Check if the last event is a drought
        if current_event_start is not None and consecutive_days >= threshold_days:
            start_date = datetime.fromisoformat(weather_data[current_event_start]["date"])
            end_date = datetime.fromisoformat(weather_data[-1]["date"])
            
            # Calculate intensity (precipitation deficit)
            event_precips = [weather_data[j]["precipitation"] for j in range(current_event_start, len(weather_data))]
            intensity = threshold - np.mean(event_precips)
            
            events.append(HazardEvent(
                start_date=start_date,
                end_date=end_date,
                duration=consecutive_days,
                intensity=round(max(0, intensity), 2),
                hazard_type="drought"
            ))
        
        return events

class TrendAnalyzer:
    @staticmethod
    def calculate_trends(events_by_year: Dict[int, List[HazardEvent]]) -> List[Dict]:
        """Calculate yearly trend statistics with enhanced metrics"""
        trends = []
        
        for year, events in events_by_year.items():
            if events:
                frequency = len(events)
                avg_intensity = np.mean([e.intensity for e in events])
                avg_duration = np.mean([e.duration for e in events])
                max_intensity = max([e.intensity for e in events])
                max_duration = max([e.duration for e in events])
                total_duration = sum([e.duration for e in events])
                
                # Enhanced seasonal distribution calculation
                winter_events = [e for e in events if e.start_date.month in [12, 1, 2]]
                spring_events = [e for e in events if e.start_date.month in [3, 4, 5]]
                summer_events = [e for e in events if e.start_date.month in [6, 7, 8]]
                fall_events = [e for e in events if e.start_date.month in [9, 10, 11]]
                
                seasonal_counts = {
                    'winter': len(winter_events),
                    'spring': len(spring_events),
                    'summer': len(summer_events),
                    'fall': len(fall_events)
                }
                
                seasonal_durations = {
                    'winter_duration': sum([e.duration for e in winter_events]),
                    'spring_duration': sum([e.duration for e in spring_events]),
                    'summer_duration': sum([e.duration for e in summer_events]),
                    'fall_duration': sum([e.duration for e in fall_events])
                }
            else:
                frequency = max_intensity = avg_duration = max_duration = total_duration = 0
                avg_intensity = 0
                seasonal_counts = {'winter': 0, 'spring': 0, 'summer': 0, 'fall': 0}
                seasonal_durations = {
                    'winter_duration': 0, 'spring_duration': 0,
                    'summer_duration': 0, 'fall_duration': 0
                }
            
            trends.append({
                'year': year,
                'frequency': frequency,
                'intensity': round(avg_intensity, 2),
                'duration': round(avg_duration, 2),
                'max_intensity': round(max_intensity, 2),
                'max_duration': max_duration,
                'total_duration': total_duration,
                **seasonal_counts,
                **seasonal_durations
            })
        
        return sorted(trends, key=lambda x: x['year'])
    
    @staticmethod
    def calculate_summary_statistics(trends: List[Dict]) -> Dict[str, Any]:
        """Enhanced summary statistics calculation"""
        if len(trends) < 2:
            return {"error": "Insufficient data for trend analysis"}
        
        frequencies = [t['frequency'] for t in trends]
        years = [t['year'] for t in trends]
        
        # Enhanced trend analysis
        slope, intercept, r_value, p_value, std_err = stats.linregress(years, frequencies)
        
        # Calculate rolling averages for trend stability
        window_size = min(5, len(frequencies))
        rolling_avg = pd.Series(frequencies).rolling(window=window_size).mean().tolist()[-1]
        
        # Calculate recent trend (last 5 years or available data)
        recent_slope = 0
        if len(frequencies) >= 2:
            recent_years = years[-5:]
            recent_freqs = frequencies[-5:]
            recent_slope, _, _, _, _ = stats.linregress(recent_years, recent_freqs)
        
        # Calculate percentage increase
        if frequencies[0] > 0:
            percentage_increase = ((frequencies[-1] - frequencies[0]) / frequencies[0]) * 100
            rolling_increase = ((rolling_avg - frequencies[0]) / frequencies[0]) * 100
        else:
            percentage_increase = 0
            rolling_increase = 0
        
        # Calculate average annual increase
        annual_increase = slope
        
        return {
            "percentage_increase": round(percentage_increase, 1),
            "rolling_increase": round(rolling_increase, 1),
            "average_increase": round(annual_increase, 2),
            "recent_trend": round(recent_slope, 2),
            "significance": round(p_value, 4),
            "trend": "Increasing" if slope > 0 else "Decreasing" if slope < 0 else "Stable",
            "r_squared": round(r_value**2, 3),
            "confidence": "High" if p_value < 0.05 else "Medium" if p_value < 0.1 else "Low",
            "trend_stability": "Stable" if abs(recent_slope - slope) < 0.5 else "Variable"
        }

# Database helper functions
async def store_weather_data(region_id: str, weather_data: List[Dict]):
    """Store weather data in MongoDB"""
    documents = []
    for data in weather_data:
        doc = {
            "region_id": region_id,
            "date": datetime.fromisoformat(data["date"]),
            "temperature_max": data["temperature_max"],
            "temperature_min": data["temperature_min"],
            "precipitation": data["precipitation"],
            "humidity": data["humidity"],
            "created_at": datetime.utcnow()
        }
        documents.append(doc)
    
    if documents:
        await weather_collection.insert_many(documents)

async def get_cached_weather_data(region_id: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    """Retrieve cached weather data from MongoDB"""
    cursor = weather_collection.find({
        "region_id": region_id,
        "date": {"$gte": start_date, "$lte": end_date}
    }).sort("date", 1)
    
    data = []
    async for doc in cursor:
        data.append({
            "date": doc["date"].isoformat(),
            "temperature_max": doc["temperature_max"],
            "temperature_min": doc["temperature_min"],
            "precipitation": doc["precipitation"],
            "humidity": doc["humidity"]
        })
    
    return data

async def store_analysis_result(result: AnalysisResult):
    """Store analysis result in MongoDB"""
    doc = result.model_dump()
    doc["created_at"] = datetime.utcnow().isoformat()  # Convert to ISO format string
    await analysis_collection.insert_one(doc)

async def get_cached_analysis(region_id: str, hazard_type: str, start_year: int, end_year: int) -> Optional[Dict]:
    """Retrieve cached analysis result"""
    doc = await analysis_collection.find_one({
        "region.lat": {"$exists": True},  # We'll need to match by coordinates
        "hazard_type": hazard_type,
        "time_range.start_year": start_year,
        "time_range.end_year": end_year,
        "created_at": {"$gte": datetime.utcnow() - timedelta(hours=24)}  # Cache for 24 hours
    })
    
    if doc:
        doc["created_at"] = doc["created_at"].isoformat()
    
    return doc

# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


manager = ConnectionManager()

# Analysis and broadcasting function
async def analyze_and_broadcast(websocket: WebSocket, request: AnalysisRequest):
    """Enhanced analysis and broadcasting function"""
    try:
        region_id = f"{request.region.lat}_{request.region.lon}"
        
        await websocket.send_json({
            "status": "fetching",
            "message": "Fetching weather data..."
        })

        start_date = f"{request.time_range.start_year}-01-01"
        end_date = f"{request.time_range.end_year}-12-31"
        
        weather_service = WeatherDataService()
        weather_data = await weather_service.fetch_historical_data(
            request.region.lat, request.region.lon, start_date, end_date
        )

        await websocket.send_json({
            "status": "analyzing",
            "message": "Analyzing climate patterns..."
        })

        # Analyze hazards with progress updates
        analyzer = HazardAnalyzer()
        if request.hazard_type == "heatwave":
            events = analyzer.detect_heatwaves(weather_data)
        elif request.hazard_type == "drought":
            events = analyzer.detect_droughts(weather_data)
        elif request.hazard_type == "heavy_rain":
            events = analyzer.detect_heavy_rain(weather_data)

        # Group events by year with progress update
        events_by_year = {}
        for year in range(request.time_range.start_year, request.time_range.end_year + 1):
            events_by_year[year] = []

        for event in events:
            year = event.start_date.year
            if year in events_by_year:
                events_by_year[year].append(event)

        await websocket.send_json({
            "status": "calculating",
            "message": "Calculating trends..."
        })

        # Calculate trends with enhanced metrics
        trend_analyzer = TrendAnalyzer()
        trends = trend_analyzer.calculate_trends(events_by_year)
        summary = trend_analyzer.calculate_summary_statistics(trends)

        # Add additional context to the result
        result = AnalysisResult(
            region=request.region,
            hazard_type=request.hazard_type,
            time_range=request.time_range,
            trends=trends,
            summary={
                **summary,
                "analysis_period": f"{request.time_range.start_year}-{request.time_range.end_year}",
                "total_events": sum(t['frequency'] for t in trends),
                "avg_events_per_year": round(np.mean([t['frequency'] for t in trends]), 1)
            },
            created_at=datetime.utcnow()
        )

        # Cache the result
        await store_analysis_result(result)

        # Send final result
        result_dict = result.model_dump()
        # Convert datetime objects to strings
        result_dict["created_at"] = result_dict["created_at"].isoformat()
        for trend in result_dict["trends"]:
            # No datetime objects in TrendData
            pass
        await websocket.send_json({
            "status": "complete",
            "data": result_dict
        })

    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        await websocket.send_json({
            "status": "error",
            "message": str(e)
        })

# WebSocket endpoint for real-time analysis
@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket):
    try:
        await manager.connect(websocket)
        
        while True:
            try:
                data = await websocket.receive_json()
                if data.get("type") == "keepalive":
                    logger.debug("Received keepalive message")
                    continue  # Ignore keepalive messages

                try:
                    request = AnalysisRequest(**data)
                except Exception as e:
                    logger.error(f"Validation error: {e}")
                    await websocket.send_json({"status": "error", "message": "Invalid request format"})
                    continue

                asyncio.create_task(analyze_and_broadcast(websocket, request))
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                # Only send error if the connection is still open
                if websocket.client_state != WebSocketState.CLOSED:
                    try:
                        await websocket.send_json({
                            "status": "error",
                            "message": str(e)
                        })
                    except Exception as send_error:
                        logger.error(f"Error sending error message: {send_error}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
