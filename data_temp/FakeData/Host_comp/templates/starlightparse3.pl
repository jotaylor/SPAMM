#!/usr/bin/perl

#  Program to take STARLIGHT output and format it for 
#  MODSfit

sub parseoutput{
    $blue_spec = $_[0];
    $err_spec = $_[1];
    $out_file = $_[2];

    print "$blue_spec \n";
    open(OUT, ">$out_file") or die "no output file specified\n";
    open(BLUES, "<$blue_spec") or die "A blue spectrum is good to have...\n";

    $i = 0;
    while(<BLUES>){
        chomp;
        ($b1,$b2,$b3,$b4,$b5) = split /\s+/,$_;
        if($i eq 1){
            $wave = $b2;
            $flux = ($b3 - 0.*$b4)*$norm;
	    $model = $b4*$norm;
            push(@spec_wave,$wave);
            push(@spec_flux,$flux);
            push(@spec_mod,$model);
        }
        if($b2 eq "[fobs_norm"){
            $norm = $b1;
        }
        if($b3 eq "[Nl_obs]"){
            $Nobs = $b2;
            $i = 1;
        }
    		
    }
    print "Blue spectrum: \n";
    print "   normalization: $norm  \n";
    print "   number of points: $Nobs \n";
    close(BLUESS);

    foreach $item (@spec_wave){
        open(ERRS, "<$err_spec") or die "An error spectrum is good to have... $err_spec\n";
        while(<ERRS>){
            chomp;
            ($e1,$e2,$e3,$e4,$e5) = split /\s+/,$_;
            if($item == $e2){
                push(@spec_err,$e4);
            }
        }
        close(ERRS);
    }

    $size = @spec_err;
    print "$size \n";
    for($i = 0; $i<=@spec_wave; $i++){
        print OUT "@spec_wave[$i] @spec_flux[$i] @spec_mod[$i]\n";
    }
    undef @spec_wave;
    undef @spec_flux;
    undef @spec_err;
    undef @spec_mod;
}
#######################
$base = $ARGV[0];
$numb = $ARGV[1];
$nume = $ARGV[2];
for($j = 1; $j<=1; $j++){
    my $bname = join('',$base,"_template.starlight");
    my $ename = join('',$base,"_template.spec");
    my $oname = join('',$base,"_template.mod");
    &parseoutput($bname,$ename,$oname);
}
