function adv = adversarial_perturbation(x,l,Df_base,f_out,opts)
if(nargin==5)
    NUM_LABELS = opts.labels_limit;
    OS = opts.overshoot;
else
    NUM_LABELS = 10;
    OS = 0.02;
end

Df = @(y,idx) Df_base(y,l,idx);

ff = f_out(x,0);
ff = ff-ff(l);
[~,I] = sort(ff,'descend');
labels = I(2:NUM_LABELS);

r = x*0;
x_u = x;

itr = 0;
while(f_out(x+(1+OS)*r,1)==l && itr<100)
    itr = itr + 1;
        
    ff = f_out(x_u,0);
    ff = ff-ff(l);
    
    idx = [l labels];
    ddf = Df(x_u,idx);
    
    dr = project_on_polyhedron(ddf,ff(idx));
       
    x_u = x_u+dr;
    r = r + dr;
end

adv.r = (1+OS)*r;
adv.new_label = f_out(x+(1+OS)*r,1);
adv.itr = itr;
end

function dir = project_on_polyhedron(Df,f)
res = abs(f)./sqrt(sum(Df.*Df));
[~,ii]=min(res);
dir = res(ii)*Df(:,ii)/norm(Df(:,ii));
end