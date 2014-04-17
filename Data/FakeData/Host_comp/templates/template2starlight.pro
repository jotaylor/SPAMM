pro template2starlight,in,out

readcol,in,wave,flux
err = flux*0.1
fflag = flux*0 + 1
writecol,out,wave,flux,err,fflag
end

