import numpy as np
from utils import mapCorrelation, bresenham2D

class MAP:
    def __init__(self, resolution, xmin, ymin, xmax, ymax) -> None:
        MAP = {}
        MAP['res']   = resolution #meters
        MAP['xmin']  = xmin  #meters
        MAP['ymin']  = ymin  
        MAP['xmax']  = xmax
        MAP['ymax']  = ymax
        MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
        MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
        MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float32) 

        self.MAP = MAP


    def get_pixel_coordinates(self, world_coordinates):

        # convert from meters to cells
        x = world_coordinates[0, :]
        y = world_coordinates[1, :]

        xis = np.ceil((x - self.MAP['xmin']) / self.MAP['res'] ).astype(np.int16)-1
        yis = np.ceil((y - self.MAP['ymin']) / self.MAP['res'] ).astype(np.int16)-1

        xis = np.clip(xis, 0, self.MAP["sizex"] - 1)
        yis = np.clip(yis, 0, self.MAP["sizey"] - 1)

        return xis, yis
    

    def update_occ_map(self, ranges_coords_world, lidar_pos):
        
        xis, yis = self.get_pixel_coordinates(ranges_coords_world)
        lidar_x, lidar_y = self.get_pixel_coordinates(lidar_pos)

        startx = lidar_x[0]
        starty = lidar_y[0]

        for x, y in zip(xis, yis):
            cells = bresenham2D(startx, starty, x, y).astype(np.int32)
            cells_x = np.clip(cells[0,:], 0, self.MAP['sizex']-1)
            cells_y = np.clip(cells[1,:], 0, self.MAP['sizey']-1)

            # Mark free as -1 all but last
            self.MAP["map"][cells_x[:-10], cells_y[:-10]] -= np.log(4)
            # Mark last as occupied
            self.MAP["map"][cells_x[-10:], cells_y[-10:]] += np.log(4)

        self.MAP["map"] = np.clip(self.MAP["map"], -100, 100)
    
    def find_correlation_with_current_map(self, lidar_range):

        x_im = np.arange(self.MAP['xmin'],self.MAP['xmax']+self.MAP['res'],self.MAP['res']) #x-positions of each pixel of the map
        y_im = np.arange(self.MAP['ymin'],self.MAP['ymax']+self.MAP['res'],self.MAP['res']) #y-positions of each pixel of the map

        x_range = np.arange(-0.2,0.2+0.05,0.05)
        y_range = np.arange(-0.2,0.2+0.05,0.05)

        binary_map = np.zeros_like(self.MAP["map"])
        binary_map[self.MAP["map"]==100] = 1
        binary_map[self.MAP["map"]==-100] = -1
        c = mapCorrelation(binary_map, x_im, y_im, lidar_range, x_range, y_range)
        return c
    
    def get_correlation_with_map(self, particles, lidar_ranges_body_frame):

        # Shape 2,2,N
        # all_rotations = np.stack([get_rotation_matrix(x[2]) for x in particles.T], axis=-1)
        
        theta = particles[2,:]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        r1 = np.stack([cos_theta, -1*sin_theta]).reshape((1,2,-1))
        r2 = np.stack([sin_theta, cos_theta]).reshape((1,2,-1))
        all_rotations = np.concatenate([r1,r2], axis=0)

        
        # Shape 2, N
        all_positions = particles[0:2, :]

        # Shape N,2,2
        all_rotations = all_rotations.transpose((2,0,1))
        # Shape N,2
        all_positions = all_positions.T

        # Shape N,2,scans
        lidar_ranges_world = all_rotations @ lidar_ranges_body_frame + all_positions[:,:,None]

        correlations = self.find_correlation_with_current_map(lidar_ranges_world)        
        return correlations


    def convert_to_color(self):
        mp = self.MAP["map"]
        color_map = np.stack([mp]*3, axis=-1)
        self.MAP["map"] = color_map

