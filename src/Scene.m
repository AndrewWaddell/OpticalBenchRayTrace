classdef Scene
    properties
        sources
        shapes
        sensors
        rays
        medium
    end
    methods
        function generate_rays()
            rays.add_rays = source.generate_rays();
        end
        function trace_high_level(obj)
            % generate rays
            % while rays.ective not empty
            % make rays.active 0 if sensor, or no int
            % trace low_res objects
            % d to POIS
            % update POI to smallest d
        end
    end
end