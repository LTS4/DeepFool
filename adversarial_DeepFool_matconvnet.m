%%
%   MATLAB code for DeepFool
%
%   adversarial_DeepFool_matconvnet(x,net):
%   computes the adversarial perturbations for a MatConvNet's model
%
%   INPUTS
%   x: image in W*H*C format
%   net: MatConvNet's network (without loss layer)
%   OUTPUTS
%   r_hat: minimum perturbation
%   l_hat: adversarial label
%   l: classified label
%   itr: number of iterations
%
%   please cite: arXiv:1511.04599
%%
function [r_hat,l_hat,l,itr] = adversarial_DeepFool_matconvnet(x,net)
size_x = size(x);
c = numel(net.layers{end}.weights{2});

x = reshape(x,numel(x),1);
l=f(x,1);

adv = adversarial_perturbation(x,l,@Df,@f);

l_hat = adv.new_label;
r_hat = adv.r;
itr = adv.itr;

    function out = f(y,flag)
        for i=1:c
            %do forward pass
            res = vl_simplenn(net,single(reshape(y,size_x)),[],[],'disableDropout',true);
            out(i) = res(end).x(i);
        end
        
        %flag==0:compute the outputs
        %flag==1:compute the label
        if flag==1
            [~,out] = max(out);
        end
    end

    function dzdx = Df(y,label,idx)
        for i=1:numel(idx)
            
            dzdy = zeros(1,1,c,'single');
            dzdy(idx(i)) = 1;
            
            %do forward-backward pass
            res = vl_simplenn(net,single(reshape(y,size_x)),dzdy,[],'disableDropout',true);
            dzdx(:,i) = reshape(res(1).dzdx,prod(size_x),1);
        end
        dzdx = dzdx-repmat(dzdx(:,idx==label),1,numel(idx));
    end

end