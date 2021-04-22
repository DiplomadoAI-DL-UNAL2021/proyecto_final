### Funciones y constantes del proyecto

import math
import folium
import ee

###################### Funciones para calcular la probabilidad de nubosidad
"""Add a "cloud" binary band to an image (0: no cloud, 1: cloud)
using QA60 info"""

def extractQABits(qaBand, bitStart, bitEnd):
    numBits = bitEnd - bitStart + 1
    qaBits = qaBand.rightShift(bitStart).mod(math.pow(2, numBits))
    return qaBits

def computeClouds(img):
    qa = img.select('QA60')
    clouds = extractQABits(qa, 10, 10)
    cirrus = extractQABits(qa, 11, 11)
    final = clouds.Or(cirrus).rename('clouds')
    return img.addBands(final)

def cloudsAtRegion(region):
    def wrap(img):
        clouds = img.select('clouds')
        val = clouds.reduceRegion(**{
          "reducer": ee.Reducer.mode(),
          "geometry": region,
          "scale": 60 # QA60 -> 60m
        }).get('clouds')
        return img.set('CLOUDS_AT_REGION', val)
    
    return wrap


def filterClouds(region):
    def wrap(img):
        try:
            clouds = img.select('probability');

            cloudiness = clouds.reduceRegion(**{
                "reducer": ee.Reducer.median(),
                "geometry": region,
                "scale": 10,
            }).get('probability')

            return img.set('CLOUDS_AT_AOI', cloudiness)
        except:
            return img.set('CLOUDS_AT_AOI', 100)

    return wrap


################## Parámetros de despliegue

# Define a method for displaying Earth Engine image tiles on a folium map.
def add_ee_layer(self, ee_object, vis_params, name):
    
    try:    
        # display ee.Image()
        if isinstance(ee_object, ee.image.Image):    
            map_id_dict = ee.Image(ee_object).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)

        # display ee.ImageCollection()
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):    
            ee_object_new = ee_object.mosaic()
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
            ).add_to(self)

        # display ee.Geometry()
        elif isinstance(ee_object, ee.geometry.Geometry):    
            folium.GeoJson(
            data = ee_object.getInfo(),
            name = name,
            overlay = True,
            control = True
        ).add_to(self)

        # display ee.FeatureCollection()
        elif isinstance(ee_object, ee.featurecollection.FeatureCollection):  
            ee_object_new = ee.Image().paint(ee_object, 0, 2)
            map_id_dict = ee.Image(ee_object_new).getMapId(vis_params)
            folium.raster_layers.TileLayer(
            tiles = map_id_dict['tile_fetcher'].url_format,
            attr = 'Google Earth Engine',
            name = name,
            overlay = True,
            control = True
        ).add_to(self)
    
    except:
        print("Could not display {}".format(name))


# Add EE drawing method to folium.
folium.Map.add_ee_layer = add_ee_layer

imageVisParam = {
  "bands": ["B4","B3","B2"],
  "gamma": 1.4000000000000001,
  "max": 4712,
  "min": 55,
  "opacity": 1
}

cloudVisParam = { 
    "bands": ["B1"],
    "max": 7116.686205039902,
    "min": 1090.0215691879994,
    "opacity": 1,
    "palette": ["106fff","ffec60","ff0000"]
}

cloud1VisParam = {
    "bands": ["probability"],
    "max": 100,
    "opacity": 1,
    "palette": ["06c014","e6ff5e","ffd04d","ff650a"]
}    



# Map.addLayer(test_image, imageVisParam, 'S2')
#Map.addLayer(masked, imageVisParam, 'masked')
#Map.addLayer(test_image, cloudVisParam, 'Whole Scene Cloudiness', True, 0.5)
#Map.addLayer(tunjo.geometry(), {}, "Tunjo", True, 0.2)
#Map.addLayer(imgS2Cloud, cloud1VisParam, 'Cloud1', True)
#Map.addLayer(region, {}, "Tunjo buffer", True, 0.7)
#Map
#imgS2Cloud.getMapId(cloud1VisParam)['tile_fetcher'].url_format