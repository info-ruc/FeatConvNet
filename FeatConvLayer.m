classdef FeatConvLayer < nnet.layer.Layer
    
    properties (Learnable)
        % Layer learnable parameters.
        % coefficient.
        Alpha
        Beta
    end
    
    methods
        function layer = FeatConvLayer(name, n)
            % Set layer name.
            layer.Name = name;
            num = combntns(n,2);
            
            % Set layer description.
            layer.Description = name;
            
            % Initialize coefficient.
            layer.Alpha = rand([num,1]);
            layer.Beta = rand([num,1]);
        end
        
        function Z = predict(layer, X)
            Z=[];counter=1;s=size(X);
            for  i = 1:s(1)
                for j = i+1:s(1)
                    Zi = layer.Alpha(counter).*X(i,:)+layer.Beta(counter).*X(j,:);
                    Z = [Z ;Zi];
                    counter = counter +1;
                end
            end
            Z = [Z; X];
        end
    end
end