function adv = adversarial_perturbation(x,l,Df_base,f_out)
NUM_LABELS = 10;

Df = @(y,idx) Df_base(y,l,idx);

ff = f_out(x,0);
ff = ff-ff(l);
[~,I] = sort(ff,'descend');
labels = I(2:NUM_LABELS);

r = x*0;
x_u = x;

itr = 0;
while(f_out(x+1.02*r,1)==l && itr<100)
    itr = itr + 1;
        
    ff = f_out(x_u,0);
    ff = ff-ff(l);
    
    idx = [l labels];
    ddf = Df(x_u,idx);
    
    dr = project_on_polyhedron(ddf,ff(idx));
       
    x_u = x_u+dr;
    r = r + dr;
end

adv.r = 1.02*r;
adv.new_label = f_out(x+1.02*r,1);
adv.itr = itr;
end
function dir = project_on_polyhedron(Df,f)
res = abs(f)./sqrt(sum(Df.*Df));
%for i=1:numel(f)
%    res(i) = abs(f(i))/norm(Df(:,i));
%end
[~,ii]=min(res);
dir = res(ii)*Df(:,ii)/norm(Df(:,ii));
end