$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false)
$state='ok'; $problem=''; $payload=@{}
try {
    switch ($probe) {
        { $_ -in 'events-system','events-application' } {
            $log=if($probe -eq 'events-system'){'System'}else{'Application'}
            $filter=@{LogName=$log;StartTime=$since}
            if($log -eq 'System'){$filter.Level=1,2,3}else{$filter.Id=1000,1002,508,510,532,533}
            $records=@(Get-WinEvent -FilterHashtable $filter -MaxEvents ($limit+1) -ErrorAction SilentlyContinue)
            if($Error.Count -and $Error[0].FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*') {
                throw $Error[0]
            }
            if($records.Count -gt $limit){$state='truncated';$problem="only the newest $limit matching records were retained"}
            $records=@($records | Select-Object -First $limit)
            if($log -eq 'Application'){
                $Error.Clear()
                $wer=@(Get-WinEvent -FilterHashtable @{LogName='Application';Id=1001;ProviderName='Windows Error Reporting';StartTime=$since} -MaxEvents ($limit+1) -ErrorAction SilentlyContinue)
                if($Error.Count -and $Error[0].FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*'){throw $Error[0]}
                if($wer.Count -gt $limit){$state='truncated';$problem="WER capped at $limit records; other application errors retained separately"}
                $records+=@($wer | Select-Object -First $limit)
                $Error.Clear()
                $checks=@(Get-WinEvent -FilterHashtable @{LogName='Application';ProviderName='Microsoft-Windows-Wininit','Chkdsk';Id=1001,26226;StartTime=$since} -MaxEvents ($limit+1) -ErrorAction SilentlyContinue)
                if($Error.Count -and $Error[0].FullyQualifiedErrorId -notlike 'NoMatchingEventsFound*'){throw $Error[0]}
                if($checks.Count -gt $limit){$state='truncated';$problem+='; disk-check records capped at '+$limit}
                $records+=@($checks | Select-Object -First $limit)
            }
            $payload.events=@($records | Sort-Object TimeCreated | ForEach-Object {
                $event=$_; $xml=[xml]$event.ToXml(); $data=@{}; $message=''
                $index=0
                foreach($item in $xml.Event.EventData.Data){
                    $name=if($item.Name){[string]$item.Name}else{[string]$index}
                    $value=if($item -is [Xml.XmlElement]){$item.InnerText}else{[string]$item}
                    $data[$name]=[string]$value; $index++
                }
                try{$message=[string]$event.FormatDescription()}catch{}
                $occurred=$null
                if($event.ProviderName -eq 'EventLog' -and $event.Id -eq 6008){
                    try{
                        $datePart=if($data.ContainsKey('param2')){$data['param2']}else{$data['1']}
                        $timePart=if($data.ContainsKey('param1')){$data['param1']}else{$data['0']}
                        $date=($datePart+' '+$timePart) -replace '\p{Cf}',''
                        $parts=($datePart -replace '\p{Cf}','') -split '/'
                        if($parts.Count -eq 3 -and [int]$parts[0] -gt 12){
                            $parsed=[datetime]::ParseExact($date,'d/M/yyyy H:mm:ss',[Globalization.CultureInfo]::InvariantCulture)
                        }elseif($parts.Count -eq 3 -and [int]$parts[1] -gt 12){
                            $parsed=[datetime]::ParseExact($date,'M/d/yyyy H:mm:ss',[Globalization.CultureInfo]::InvariantCulture)
                        }else{
                            $parsed=[datetime]::Parse($date,[Globalization.CultureInfo]::CurrentCulture)
                        }
                        $occurred=$parsed.ToUniversalTime().ToString('o')
                    }catch{}
                }
                @{
                    occurred_at=$occurred;xml=$event.ToXml()
                    log=$log;provider=[string]$event.ProviderName;id=[int]$event.Id
                    record_id=[long]$event.RecordId;reported_at=$event.TimeCreated.ToUniversalTime().ToString('o')
                    message=$message;data=$data
                }
            })
        }
        'inventory' {
            $os=Get-CimInstance Win32_OperatingSystem
            $bios=Get-CimInstance Win32_BIOS
            $version=Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
            $payload.os=@{caption=$os.Caption;build=$os.BuildNumber;revision=$version.UBR;display_version=$version.DisplayVersion;last_boot=$os.LastBootUpTime.ToUniversalTime().ToString('o')}
            $payload.bios=@{version=$bios.SMBIOSBIOSVersion;released=$bios.ReleaseDate.ToUniversalTime().ToString('o')}
            $payload.updates=@(Get-HotFix | Select-Object HotFixID,Description,InstalledOn)
            $payload.antivirus=@(Get-CimInstance -Namespace root/SecurityCenter2 -ClassName AntivirusProduct | Select-Object displayName,productState,pathToSignedProductExe)
            $payload.defender=Get-MpComputerStatus | Select-Object AMRunningMode,AntivirusEnabled,RealTimeProtectionEnabled,IsTamperProtected
            $payload.drivers=@(Get-CimInstance Win32_PnPSignedDriver | Select-Object DeviceID,DeviceName,DriverVersion,DriverDate,DriverProviderName,InfName)
            $payload.services=@(Get-CimInstance Win32_Service | Select-Object Name,DisplayName,State,StartMode,PathName)
            $payload.service_registry=@(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Services\*' | Select-Object PSChildName,DisplayName,ImagePath,Start,Type,DependOnService)
            $payload.system_drivers=@(Get-CimInstance Win32_SystemDriver | Where-Object {$_.State -ne 'Running' -and $_.StartMode -in 'Auto','Boot','System'} | Select-Object Name,State,StartMode,PathName)
            $memory=Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
            $payload.memory=@{available_bytes=$memory.AvailableBytes;committed_bytes=$memory.CommittedBytes;commit_limit=$memory.CommitLimit;sampled_at=(Get-Date).ToUniversalTime().ToString('o')}
            $payload.pagefiles=@(Get-CimInstance Win32_PageFileUsage | Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage)
        }
        'storage' {
            $payload.disks=@(Get-Disk | Select-Object Number,FriendlyName,SerialNumber,FirmwareVersion,BusType,HealthStatus,OperationalStatus,IsBoot,IsSystem,Size,UniqueId)
            $payload.partitions=@(Get-Partition | Select-Object DiskNumber,PartitionNumber,DriveLetter,Type,GptType,IsSystem,IsBoot,Size,AccessPaths)
            $payload.volumes=@(Get-Volume | Select-Object DriveLetter,FileSystem,HealthStatus,OperationalStatus,Size,SizeRemaining,Path)
            $payload.reliability=@(Get-PhysicalDisk | ForEach-Object {
                $disk=$_; $counters=$null; $errorText=''
                try{$counters=$disk | Get-StorageReliabilityCounter -ErrorAction Stop | Select-Object Temperature,TemperatureMax,Wear,ReadErrorsTotal,ReadErrorsUncorrected,WriteErrorsTotal,WriteErrorsUncorrected,ReadLatencyMax,WriteLatencyMax,FlushLatencyMax,PowerOnHours}catch{$errorText=$_.Exception.Message}
                @{device_id=$disk.DeviceId;name=$disk.FriendlyName;serial=$disk.SerialNumber;health=[string]$disk.HealthStatus;counters=$counters;error=$errorText}
            })
        }
        'network' {
            $payload.adapters=@(Get-NetAdapter -Physical | ForEach-Object {
                @{name=$_.Name;description=$_.InterfaceDescription;status=[string]$_.Status;interface_index=$_.ifIndex;wireless=([int]$_.NdisPhysicalMedium -in 1,9);pnp_id=$_.PnPDeviceID;driver_version=$_.DriverVersion;link_speed=$_.LinkSpeed}
            })
            $payload.routes=@(Get-NetRoute -AddressFamily IPv4 -DestinationPrefix '0.0.0.0/0' | Select-Object InterfaceIndex,InterfaceAlias,NextHop,RouteMetric,State)
        }
        'devices' {
            $payload.devices=@(Get-PnpDevice -PresentOnly | Where-Object {$_.InstanceId -like 'PCI\*'} | ForEach-Object {
                $device=$_; $properties=@{}; $propertyError=''
                try{
                    Get-PnpDeviceProperty -InstanceId $device.InstanceId -KeyName 'DEVPKEY_Device_Parent','DEVPKEY_Device_LocationPaths','DEVPKEY_Device_HardwareIds','DEVPKEY_Device_LocationInfo' -ErrorAction Stop | ForEach-Object {$properties[$_.KeyName]=$_.Data}
                }catch{$propertyError=$_.Exception.Message}
                @{id=$device.InstanceId;name=$device.FriendlyName;status=[string]$device.Status;class=$device.Class;properties=$properties;error=$propertyError}
            })
        }
        'dumps' {
            $errors=@(); $files=@()
            foreach($directory in @("$env:WINDIR\LiveKernelReports","$env:WINDIR\Minidump")){
                try{
                    $files+=@(Get-ChildItem -LiteralPath $directory -Filter '*.dmp' -Recurse -File -ErrorAction Stop | Select-Object FullName,Length,@{n='written_at';e={$_.LastWriteTime.ToUniversalTime().ToString('o')}})
                }catch{if($_.CategoryInfo.Category -ne 'ObjectNotFound'){$errors+=$_.Exception.Message}}
            }
            $payload.dumps=@($files | Sort-Object written_at -Descending | Select-Object -First $limit)
            if($errors.Count){$state='unavailable';$problem=$errors -join '; '}
            if($files.Count -gt $limit){$state='truncated';$problem="only the newest $limit dump records were retained"}
        }
        'integrity' {
            $identity=[Security.Principal.WindowsIdentity]::GetCurrent()
            $principal=[Security.Principal.WindowsPrincipal]::new($identity)
            if(-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)){
                $state='unavailable';$problem='component-store health requires an elevated terminal'
            }else{
                $payload.text=(& "$env:WINDIR\System32\dism.exe" /Online /Cleanup-Image /CheckHealth /English | Out-String)
                $payload.exit_code=$LASTEXITCODE
                if($LASTEXITCODE -ne 0){$state='failed';$problem="DISM health query exited $LASTEXITCODE"}
            }
            $payload.filesystems=@(Get-CimInstance Win32_Volume | Select-Object DriveLetter,FileSystem,DeviceID,DirtyBitSet)
            $payload.boot_checks=@((Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager').BootExecute)
            $payload.reboot_pending=(Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending') -or (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired')
        }
        default {throw "Unknown diagnostic probe: $probe"}
    }
}catch{$state='failed';$problem=$_.Exception.Message}
@{name=$probe;state=$state;collected_at=(Get-Date).ToUniversalTime().ToString('o');payload=$payload;error=$problem} | ConvertTo-Json -Depth 12 -Compress
