# IDUN LSL and BCI-essentials test

## Notes

**2024-01-19**
- The objective of today's task was to get the Udin Guardian to work with BCI-essentials-Python (i.e., to stream to LSL).
- The tutorial I used can be found in the [udin-docs](https://docs.idunguardian.com/en/page-6iii-about-pysdk) website
- Installed the udin-client-guardian in a conda environment using 
```
    pip install udin-client-guardian
```
- I copied a few Python files from their [examples](https://github.com/iduntech/idun-guardian-client-examples/tree/main) repo.
    - `search`: Searches for the Udin Guardian device connected through Bluetooth.
    - `lsl_recording`: starts an LSL stream. Needs the API key, that was already requested but we do not have it.
    - `lsl_utils`: needed to run the `lsl_recording` file.
- I was able to see the LSL stream from the LSL LabRecorder, however, since there is no API key, I think that the file recorded does not contain any actual EEG data (I have not confirmed this).

**To do**
- Once we get the API key and we confirmed that the LSL stream actually has EEG data, it should only be a matter of starting BCI-essentials to get it to work.
