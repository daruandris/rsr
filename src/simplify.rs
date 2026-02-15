use std::{ffi::{CStr, CString}, os::raw::c_char};


unsafe extern "C" {
    fn simplify_symengine_cpp(input: *const c_char, output: *mut c_char, max_len: usize);
}

pub fn simplify_symengine(eq: &str) -> String {
    let c_input = CString::new(eq).unwrap_or_else(|_| CString::new("").unwrap());
    let mut buffer = vec![0u8; 1024];
    unsafe {
        simplify_symengine_cpp(
            c_input.as_ptr(), 
            buffer.as_mut_ptr() as *mut c_char, 
            buffer.len());
        
        CStr::from_ptr(buffer.as_ptr() as *const c_char)
            .to_string_lossy()
            .into_owned()
    }
}