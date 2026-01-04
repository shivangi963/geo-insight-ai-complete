class Location:
    def __init__(self, address: str, latitude: float, longitude: float):
        self.address = address
        self.latitude = latitude
        self.longitude = longitude

    def to_dict(self):
        """Return the location details as a dictionary"""
        return {"address": self.address, "lat": self.latitude, "lon": self.longitude}

    def __repr__(self):
        return f"Location(address={self.address!r}, lat={self.latitude}, lon={self.longitude})"

if __name__ == "__main__":
    loc = Location("123 Main Street, Bangalore", 12.9716, 77.5946)
    loc2 = Location("MG Road, Bangalore", 12.975, 77.606)
    print(loc)
    print(loc2)
    print(loc.to_dict())