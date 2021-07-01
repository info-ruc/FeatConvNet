classdef FeatConvLayer < nnet.layer.Layer
    % Example custom PReLU layer.
    
    properties (Learnable)
        % Layer learnable parameters.
        
        % Scaling coefficient.
        Alpha
        Beta
        %Delta
    end
    
    methods
        function layer = FeatConvLayer(name, n)
            % layer = preluLayer(numChannels, name) creates a PReLU layer
            % with numChannels channels and specifies the layer name.
            
            % Set layer name.
            layer.Name = name;
            num = combntns(n,2);
            % Set layer description.
            %layer.Description = "PReLU with " + numChannels + " channels";
            
            % Initialize scaling coefficient.
            %layer.Alpha = rand([1 1 numChannels]);
            %layer.Beta = rand([1 1 numChannels]);
            %layer.Delta = rand([numChannels numChannels numChannels]);
            
            layer.Alpha = rand([num,1]);
            layer.Beta = rand([num,1]);
        end
        
        function Z = predict(layer, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            
            %%Z = max(0, X) + layer.Alpha .* min(0, X);
            %Z = kron(layer.Alpha.*X,layer.Beta.*X)
            %Zi = max(0, X) + layer.Alpha .* min(0, X);
            %trueX = extractdata(X);
            %allX = combntns(trueX,2)
            %%
            Z=[];counter=1;s=size(X);
            for  i = 1:s(1)
                for j = i+1:s(1)
                    Zi = layer.Alpha(counter).*X(i,:)+layer.Beta(counter).*X(j,:);
                    Z = [Z ;Zi];
                    counter = counter +1;
                end
            end
            Z = [Z; X];
            %Z= Z';
            %%
            %Z = [X ;X;2*X;X.*X; X];
            
        end
    end
end